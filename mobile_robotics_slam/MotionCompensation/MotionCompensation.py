import numpy as np

class LaserMotionCompensator():
        def __init__(self):
                self.laser_x_pos = -0.109
                self.laser_y_pos = 0.0
                self.laser_frequency = 9.915

                self.rot_correction = 0.014 # rad
                
                self.velocity_estimate = np.zeros(3) # (x,y,theta)
                self.timestamp = 1e-6
                self.previous_velocity_estimate = np.zeros(3) # (x,y,theta)
                self.previous_timestamp = 0.0
                self.acceleration_estimate = np.zeros(3) # (x,y,theta)

        def base_to_laser_velocities(self, x_vel, y_vel, rot_vel):
                '''Converts velocities from base frame to laser frame'''
                rot_vel = rot_vel + self.rot_correction
                x_vel_lidar = x_vel -self.laser_y_pos * rot_vel
                y_vel_lidar = y_vel +self.laser_x_pos * rot_vel
                rot_vel_lidar = rot_vel 

                return np.array([x_vel_lidar, y_vel_lidar, rot_vel_lidar])

        def base_to_laser_accelerations(self, odometry_timestamp):
                '''Converts accelerations from base frame to laser frame'''
                t = odometry_timestamp.seconds + odometry_timestamp.nanoseconds * 1e-9
                dt = t - self.previous_timestamp
                self.previous_timestamp = t.copy()

                x_acc_lidar = (self.velocity_estimate[0] - self.previous_velocity_estimate[0]) / dt
                y_acc_lidar = (self.velocity_estimate[1] - self.previous_velocity_estimate[1]) / dt
                rot_acc_lidar = (self.velocity_estimate[2] - self.previous_velocity_estimate[2]) / dt

                # x_acc_lidar = x_acc - self.laser_y_pos * rot_acc
                # y_acc_lidar = y_acc + self.laser_x_pos * rot_acc
                # rot_acc_lidar = rot_acc  

                return np.array([x_acc_lidar, y_acc_lidar, rot_acc_lidar])
        
        
        def motion_compensation_pointcloud(self, laser_scan, odometry):
                laser_n_points = len(laser_scan.ranges)
                angles = laser_scan.angle_min + np.arange(laser_n_points) * laser_scan.angle_increment
                intensity = laser_scan.intensities
        
                time_i = np.arange(laser_n_points) / (self.laser_frequency*laser_n_points) # Time Interval Between Start Of Scan and Scan of Point i

                self.velocity_estimate = self.base_to_laser_velocities(odometry.twist.twist.linear.x, odometry.twist.twist.linear.y, odometry.twist.twist.angular.z)
                self.acceleration_estimate = self.base_to_laser_accelerations(odometry.timestamp)
                
                x_correction = - (self.velocity_estimate[0] * time_i + 0.5 * self.acceleration_estimate[0] * (time_i**2))
                y_correction = - (self.velocity_estimate[1] * time_i + 0.5 * self.acceleration_estimate[1] * (time_i**2))
                angle_correction = - (self.velocity_estimate[2] * time_i + 0.5 * self.acceleration_estimate[2] * (time_i**2))
        
                # Apply translations
                x = laser_scan.ranges * np.cos(angles+angle_correction) + x_correction
                y = laser_scan.ranges * np.sin(angles+angle_correction) + y_correction

                self.previous_velocity_estimate = self.velocity_estimate.copy()

                return np.column_stack((x, y, intensity))


# def laser_msg_to_pointcloud_np(self, msg):
#         '''Converts laser scan message to pointcloud'''
 
#         angle_min = msg.angle_min
#         angle_increment = msg.angle_increment
#         ranges = msg.ranges
#         intensity = msg.intensities
 
#         angles = angle_min + np.arange(len(ranges)) * angle_increment
#         cos_angles = np.cos(angles)
#         sin_angles = np.sin(angles)
#         x = ranges * cos_angles
#         y = ranges * sin_angles
#         z = np.zeros_like(ranges)
 
#         points = np.array([x, y, z, intensity]).T
 
#         return points

# def motion_compensation_pointcloud(self, pointcloud2, laser, x_vel, y_vel, rot_vel, x_acc, y_acc, rot_acc):
#         '''Algorithm from Qi Song paper'''
#         t = 1 / laser.laser_frequency
 
#         #convert vel from odom to lidar frame (https://physics.stackexchange.com/questions/197009/transform-velocities-from-one-frame-to-an-other-within-a-rigid-body)
#         x_vel_lidar = x_vel + (laser.y_pos * np.cos(laser.theta_pos) + laser.x_pos * np.sin(laser.theta_pos)) * rot_vel
#         y_vel_lidar = y_vel + (laser.y_pos * np.sin(laser.theta_pos) - laser.x_pos * np.cos(laser.theta_pos)) * rot_vel
#         rot_vel_lidar = rot_vel
 
#         x_acc_lidar = x_acc + (laser.y_pos * np.cos(laser.theta_pos) + laser.x_pos * np.sin(laser.theta_pos)) * rot_acc
#         y_acc_lidar = y_acc + (laser.y_pos * np.sin(laser.theta_pos) - laser.x_pos * np.cos(laser.theta_pos)) * rot_acc
#         rot_acc_lidar = rot_acc
            
#         weight = ((laser.laser_n_points - np.arange(pointcloud2.shape[0])) / laser.laser_n_points)
 
#         t2 = t**2
 
#         x_correction = - (x_vel_lidar * t * weight + 0.5 * x_acc_lidar * (t2) * weight)
#         y_correction = - (y_vel_lidar * t * weight + 0.5 * y_acc_lidar * (t2) * weight)
#         angle_correction = - (rot_vel_lidar * t * weight + 0.5 * rot_acc_lidar * (t2) * weight)
 
#         # Apply translations
#         translated_points = pointcloud2[:,:2] + np.column_stack((x_correction, y_correction))
 
#         # Prepare rotation matrices
#         cos_angles = np.cos(angle_correction)
#         sin_angles = np.sin(angle_correction)
 
#         # Apply rotations
#         rotated_x = translated_points[:, 0] * cos_angles - translated_points[:, 1] * sin_angles
#         rotated_y = translated_points[:, 0] * sin_angles + translated_points[:, 1] * cos_angles
 
#         # Update the pointcloud2 array in place
#         pointcloud2[:, 0] = rotated_x
#         pointcloud2[:, 1] = rotated_y
            
#         return pointcloud2
