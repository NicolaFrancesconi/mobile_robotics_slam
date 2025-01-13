import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from queue import Empty
import imageio
import os

class DynamicMapUpdater:
    def __init__(self):
        self.data_queue = multiprocessing.Queue()
        self.process = multiprocessing.Process(target=self._update_map, args=(self.data_queue,))
        self.process.daemon = True   # Ensures the thread closes with the main program
        self.update_interval = 0.1  # Update every 100ms
        self.frames_dir = "frames"  # Directory to save frames
        self.gif_path = "dynamic_map_no_icp.gif"  # Path for the final GIF

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
        while self.data_queue.qsize() >= 4:
            try:
                self.data_queue.get_nowait()  # Remove the oldest data
            except Empty:
                break  # If queue is empty, exit the loop

        # Add new data
        self.data_queue.put((poses, landmarks, points))
        

    def _update_map(self, data_queue):
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        
        angles = np.linspace(-np.pi, np.pi, 4000)
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
                        x = pose[0] + range * np.cos(angles + pose[2])
                        y = pose[1] + range * np.sin(angles + pose[2])
                        x, y = x[range < 9], y[range < 9]
                        map.extend(np.vstack((x, y)).T)
                    map = np.array(map)
                    ax.scatter(map[:, 0], map[:, 1], c='g', s=1)
                        


                # Plot landmarks
                if landmarks is not None and len(landmarks) > 0:
                    landmarks = np.array(landmarks)
                    ax.scatter(landmarks[:, 0], landmarks[:, 1], c="r", label="Landmarks")

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

        plt.ioff()
        plt.show()

    def _generate_gif(self):
        # Create GIF from saved frames
        frames = []
        frame_files = sorted([os.path.join(self.frames_dir, f) for f in os.listdir(self.frames_dir) if f.endswith(".png")])
        for frame_file in frame_files:
            frames.append(imageio.v3.imread(frame_file))

        if frames:
            imageio.mimsave(self.gif_path, frames, duration=self.update_interval)
            print(f"GIF saved to {self.gif_path}")
        else:
            print("No frames to create a GIF.")