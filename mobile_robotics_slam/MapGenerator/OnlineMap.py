import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from queue import Empty
import imageio
import os

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
                os.remove(os.path.join(self.frames_dir, f))

    def start(self):
        self.process.start()

    def stop(self):
        if self.process.is_alive():
            self.process.terminate()
            self.process.join()
            plt.close()


    def add_data(self, poses, landmarks, points):
        while self.data_queue.qsize() >= 2:
            try:
                self.data_queue.get_nowait()  # Remove the oldest data
            except Empty:
                break  # If queue is empty, exit the loop

        # Add new data
        self.data_queue.put((poses, landmarks, points))
        

    def _update_map(self, data_queue):
        #plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        
        
        frame_count = 0
        while True:
            try:
                # Retrieve latest data
                poses, landmarks, ranges = data_queue.get(timeout=self.update_interval)
                
                ax.clear()
                print("Updating Map")

                poses = np.array(poses)
                landmarks = np.array(landmarks)
                ranges = np.array(ranges)

                
                # Plot poses
                map = []
                if poses is not None and len(poses) > 0:
                    poses = np.array(poses)
                    ax.plot(poses[:, 0], poses[:, 1], "orange", label='Optimized Trajectory')
                    for pose, range in zip(poses, ranges):
                        angles = np.linspace(-np.pi, np.pi, len(range))
                        x = pose[0] + range * np.cos(angles + pose[2])
                        y = pose[1] + range * np.sin(angles + pose[2])
                        x, y = x[range < 5], y[range < 5]
                        map.extend(np.vstack((x, y)).T)
                    map = np.array(map)
                    ax.scatter(map[:, 0], map[:, 1], c='g', s=1)
                        


                # Plot landmarks
                if landmarks is not None and len(landmarks) > 0:
                    landmarks = np.array(landmarks)
                    ax.scatter(landmarks[:, 0], landmarks[:, 1], c="r", label="Corners", s=4)

                ax.set_title("Dynamic Map")
                ax.set_aspect('equal')
                ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
                print("Map Updated")
                frame_path = os.path.join(self.frames_dir, f"frame_{frame_count:04d}.png")
                plt.savefig(frame_path)
                frame_count += 1
                plt.pause(0.2)

            except Empty:
                # Handle queue timeout
                continue

            except Exception as e:
                print(e)
                # Handle queue timeout or other exceptions
                continue
