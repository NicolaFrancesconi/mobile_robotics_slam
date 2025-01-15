import numpy as np

class LaserMotionCompensator():
        def __init__(self):
                self.velocity_estimate = np.zeros(3) # (x,y,theta)
                self.previous_velocity_estimate = np.zeros(3) # (x,y,theta)
                self.acceleration_estimate = np.zeros(3) # (x,y,theta)
        




def laser_msg_to_pointcloud_np(self, msg):
        '''Converts laser scan message to pointcloud'''
 
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        ranges = msg.ranges
        intensity = msg.intensities
 
        angles = angle_min + np.arange(len(ranges)) * angle_increment
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        x = ranges * cos_angles
        y = ranges * sin_angles
        z = np.zeros_like(ranges)
 
        points = np.array([x, y, z, intensity]).T
 
        return points

def motion_compensation_pointcloud(self, pointcloud2, laser, x_vel, y_vel, rot_vel, x_acc, y_acc, rot_acc):
        '''Algorithm from Qi Song paper'''
        t = 1 / laser.laser_frequency
 
        #convert vel from odom to lidar frame (https://physics.stackexchange.com/questions/197009/transform-velocities-from-one-frame-to-an-other-within-a-rigid-body)
        x_vel_lidar = x_vel + (laser.y_pos * np.cos(laser.theta_pos) + laser.x_pos * np.sin(laser.theta_pos)) * rot_vel
        y_vel_lidar = y_vel + (laser.y_pos * np.sin(laser.theta_pos) - laser.x_pos * np.cos(laser.theta_pos)) * rot_vel
        rot_vel_lidar = rot_vel
 
        x_acc_lidar = x_acc + (laser.y_pos * np.cos(laser.theta_pos) + laser.x_pos * np.sin(laser.theta_pos)) * rot_acc
        y_acc_lidar = y_acc + (laser.y_pos * np.sin(laser.theta_pos) - laser.x_pos * np.cos(laser.theta_pos)) * rot_acc
        rot_acc_lidar = rot_acc
            
        weight = ((laser.laser_n_points - np.arange(pointcloud2.shape[0])) / laser.laser_n_points)
 
        t2 = t**2
 
        x_correction = - (x_vel_lidar * t * weight + 0.5 * x_acc_lidar * (t2) * weight)
        y_correction = - (y_vel_lidar * t * weight + 0.5 * y_acc_lidar * (t2) * weight)
        angle_correction = - (rot_vel_lidar * t * weight + 0.5 * rot_acc_lidar * (t2) * weight)
 
        # Apply translations
        translated_points = pointcloud2[:,:2] + np.column_stack((x_correction, y_correction))
 
        # Prepare rotation matrices
        cos_angles = np.cos(angle_correction)
        sin_angles = np.sin(angle_correction)
 
        # Apply rotations
        rotated_x = translated_points[:, 0] * cos_angles - translated_points[:, 1] * sin_angles
        rotated_y = translated_points[:, 0] * sin_angles + translated_points[:, 1] * cos_angles
 
        # Update the pointcloud2 array in place
        pointcloud2[:, 0] = rotated_x
        pointcloud2[:, 1] = rotated_y
            
        return pointcloud2