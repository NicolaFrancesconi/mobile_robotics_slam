import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from queue import Empty
import os
import sys

class DynamicMapUpdater:
    def __init__(self):
        print("DYNAMIC MAP STARTED")
        self.data_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=self._update_map, args=(self.data_queue,))
        self.process.daemon = True   # Ensures the thread closes with the main program
        self.update_interval = 2  # Update every 100ms
        
        path = __file__
        file_location_subfolders = 3 #Number of folder to go up to reach root of package
        for _ in range(file_location_subfolders):
            path = os.path.dirname(path)

        
        self.frames_dir = os.path.join(path, "frames")
        
        # Create frames directory if it doesn't exist
        if not os.path.exists(self.frames_dir):
            os.makedirs(self.frames_dir)
        else: # Clear existing frames
            for f in os.listdir(self.frames_dir):
                if f.endswith(".png"):
                    os.remove(os.path.join(self.frames_dir, f))

    def start(self):
        self.process.start()

    def stop(self):
        self.data_queue.close()  # Close the queue
        self.data_queue.join_thread()  # Join the thread associated with the queue
        self.process.terminate()  # Terminate the process
        self.process.join()  # Ensure the process is cleaned up
        plt.close()


    def add_data(self, poses, landmarks, points):
        while self.data_queue.qsize() >= 2:
            try:
                self.data_queue.get_nowait()  # Remove the oldest data
            except Empty:
                continue  # If queue is empty, exit the loop

            except KeyboardInterrupt:
                print("Shutting down dynamic map updater.")
                sys.exit(0)

        # Add new data
        self.data_queue.put((poses, landmarks, points))
        

    def _update_map(self, data_queue):
        #plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        
        
        frame_count = 0
        while True:
            try:
                # Retrieve latest data
                poses, landmarks, points = data_queue.get(timeout=self.update_interval)
                
                ax.clear()
                print("Updating Map")

                poses = np.array(poses)
                landmarks = np.array(landmarks)
                points = np.array(points)


                
                # Plot poses
                map = []
                if poses is not None and len(poses) > 0:
                    ax.plot(poses[:, 0], poses[:, 1], "orange", label='Optimized Trajectory')
                    x, y = points[:, 0], points[:, 1]
                    ax.scatter(x, y, c='g', s=1)

                # Plot landmarks
                if landmarks is not None and len(landmarks) > 0:
                    landmarks = np.array(landmarks)
                    ax.scatter(landmarks[:, 0], landmarks[:, 1], c="r", label="Landmarks", s=5)

                ax.set_title("Dynamic Map")
                ax.set_aspect('equal')
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
                print("Map Updated")
                frame_path = os.path.join(self.frames_dir, f"frame_{frame_count:04d}.png")
                plt.savefig(frame_path)
                frame_count += 1
                plt.pause(0.5)
                

            except Empty:
                pass

            except KeyboardInterrupt:
                pass
            finally:
                pass