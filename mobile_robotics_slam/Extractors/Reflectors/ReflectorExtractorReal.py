import numpy as np
import math

from scipy.optimize import curve_fit

class Reflector:
    def __init__(self, x_center, y_center):
        self.x = x_center
        self.y = y_center
    
    def get_position(self):
        return np.array([self.x, self.y])

class CircularMarker():
        
    def __init__(self) :
        
        # Diameter recommended by SICK = 80 mm

        # reflector_radius = 0.025
        # cluster_dist = reflector_radius + 0.02
        # reflector_min_dist = 1.0
        self.reflector_list = []
        self.reflector_radius = 0.025
        self.reflector_min_dist = 1.0
        self.cluster_dist = self.reflector_radius + 0.02
        self.reflector_intensity_thres = 150

    def laser_msg_to_pointcloud_np(self, msg, robot_pose):
        '''Converts laser scan message to pointcloud'''
 
        angle_min = msg.angle_min
        angle_increment = msg.angle_increment
        ranges = msg.ranges
        intensity = msg.intensities

 
        angles = angle_min + np.arange(len(ranges)) * angle_increment + robot_pose[2]
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        x = ranges * cos_angles + robot_pose[0]
        y = ranges * sin_angles + robot_pose[1]
        z = np.zeros_like(ranges)
        
        points = np.array([x, y, z, intensity]).T
 
        return points

    def cart_to_polar(self,points_cart):
        '''Convert cartesian coordinates to polar coordinates'''
        return np.sqrt(points_cart[0]**2 + points_cart[1]**2), np.arctan2(points_cart[1], points_cart[0])
    
    def gaussian(self, x, a, b, c):
        return a * np.exp(-((x - b)**2) / (2 * c**2))
    
    def select_centered_points(self, polar_coords, center_angle):
        
        # Find the number of points to select
        radii, angles = polar_coords
        num_points = angles.size
        num_to_select = int(np.ceil(0.5 * num_points))     #take 50% of points
        
        # Find the index of the closest angle to the center_angle
        closest_index = np.argmin(np.abs(angles - center_angle))
        
        # Determine the range of indices to select
        start_index = (closest_index - num_to_select // 2) % num_points     #TODO: try to remove %
        end_index = (start_index + num_to_select) % num_points
        
        if start_index < end_index:
            selected_indices = np.arange(start_index, end_index)
        else:
            selected_indices = np.concatenate((np.arange(start_index, num_points), np.arange(0, end_index)))    #TODO: check if else is used
        
        selected_radii = radii[selected_indices]
        selected_angles = angles[selected_indices]
        
        selected_points = [selected_radii, selected_angles]
        return selected_points
    
    def find_circle_center_with_radius_gauss(self, points):
        '''Find the centre of a circle given points intensity and using gaussian fitting'''

        # Extract intensity from the points
        intensity = points[2]

        # Convert to polar coordinates
        distances, angles = self.cart_to_polar(points[:2])

        # Perform the Gaussian fit
        try:
            popt, pcov = curve_fit(self.gaussian, angles, intensity, p0=[max(intensity), np.mean(angles), np.std(angles)])
            # Extract the parameters
            a, b, c = popt

            # Polar coordinates of the center
            selected_values = self.select_centered_points([distances, angles], b)            
            r = np.mean(selected_values[0])
            peak_theta = b      # The peak of the Gaussian is at x = b

            #PLOT (theta, intensity)
            # # Create an array of angles for plotting the Gaussian
            # theta_fit = np.linspace(min(angles), max(angles), 500)

            # # Calculate the fitted Gaussian
            # intensity_fit = self.gaussian(theta_fit, a, b, c)

            # # Plotting the points and the Gaussian fit
            # plt.figure()
            # plt.scatter(angles, intensity, label='Data Points', color='blue')
            # plt.scatter(selected_values[1], intensity[np.isin(angles, selected_values[1])], label='Selected Points', color='green')
            # plt.plot(theta_fit, intensity_fit, label='Fitted Gaussian', color='red')  
            # plt.axvline(x=peak_theta, color='green', linestyle='--', label=r'Peak at $\theta$='+f'{peak_theta:.3f}')  
            # plt.xlabel(r'$\theta$ (Angle)')
            # plt.ylabel('Intensity')
            # plt.legend()
            # plt.show()
        
        except RuntimeError:
            # Take the argmax of the intensity to find the point with the highest intensity
            print('Gaussian fitting failed. Using argmax of intensity to find the center.')
            max_intensity_index = np.argmax(intensity)
            max_intensity_angle = angles[max_intensity_index]

            selected_values = self.select_centered_points([distances, angles], max_intensity_angle)

            #Polar coordinates
            r = np.mean(selected_values[0])
            peak_theta = max_intensity_angle      
        

        centre_x = (r + self.reflector_radius) * np.cos(peak_theta)
        centre_y = (r + self.reflector_radius) * np.sin(peak_theta)

        return centre_x, centre_y
    
    def reflector_centre_gauss(self, pointcloud2, min_n_points):   #TODO: improve performance with profiler
        '''Find the centres of the reflectors given the laser points'''
        #TODO: check if is okay to keep all points in polar coordinates before motion compensation (Cartesian)
        reflector = [[],[],[]]
        reflector_points_cart = []
        first_point_is_reflector = False    #for 360° scan
        self.reflector_list = []

        raw_reflector_mask = pointcloud2[:,3] >= self.reflector_intensity_thres

        #clustering
        for i in range(pointcloud2.shape[0]):
            if raw_reflector_mask.item(i):    #.item() is faster than [i]
                if i==0:
                    first_point_is_reflector = True
                if not reflector[0]:    # if reflector is empty
                    reflector[0].append(pointcloud2.item(i,0))
                    reflector[1].append(pointcloud2.item(i,1))
                    reflector[2].append(pointcloud2.item(i,3))                
                elif math.sqrt((pointcloud2.item(i,0) - pointcloud2.item(i-1,0))**2 + (pointcloud2.item(i,1) - pointcloud2.item(i-1,1))**2) <= self.cluster_dist: #TODO: add check on angle difference referred to distance
                    reflector[0].append(pointcloud2.item(i,0))
                    reflector[1].append(pointcloud2.item(i,1))
                    reflector[2].append(pointcloud2.item(i,3))  
                    if i == pointcloud2.shape[0] - 1:  # consider if the last point is part of a reflector and compare it with the first point
                        if first_point_is_reflector and math.sqrt((pointcloud2.item(-1,0) - reflector_points_cart[0][0][0])**2 + (pointcloud2.item(-1,1) - reflector_points_cart[0][1][0])**2) <= self.cluster_dist: #TODO: add check on angle difference referred to distance
                            old = reflector_points_cart.pop(0)
                            to_add = [[],[],[]]
                            to_add[0] = reflector[0] + old[0]  
                            to_add[1] = reflector[1] + old[1]
                            to_add[2] = reflector[2] + old[2]
                            reflector_points_cart.insert(0,to_add)  # unify reflector points with the ones of the first part of scan
                        else:
                            reflector_points_cart.append(reflector)
                else:
                    reflector_points_cart.append(reflector)
                    reflector = [[],[],[]]
                    reflector[0].append(pointcloud2.item(i,0))
                    reflector[1].append(pointcloud2.item(i,1))
                    reflector[2].append(pointcloud2.item(i,3))  
            
            elif reflector[0]:  # if reflector is not empty
                    reflector_points_cart.append(reflector)
                    reflector = [[],[],[]]

        # Remove reflectors with less than min_n_points points
        checked_points_cart = [np.array(ref) for ref in reflector_points_cart if np.size(ref[0]) >= min_n_points]

        # Remove reflectors closer than reflector_min_dist
        checked_points_cart = [ref for ref in checked_points_cart if np.sqrt(ref[0][0]**2 + ref[1][0]**2) >= self.reflector_min_dist]   #TODO: check if this is correct, should be done on the centre of the reflector

        number_points = [np.size(ref[0]) for ref in checked_points_cart]

        # Find the centre of the reflectors
        ref_centres = np.zeros((2, len(checked_points_cart)))

        for i,ref in enumerate(checked_points_cart):
            ref_centres[0][i], ref_centres[1][i] = self.find_circle_center_with_radius_gauss(ref)

        x, y = ref_centres
        for x,y in zip (x,y):
            self.reflector_list.append(Reflector(x, y))

        return self.reflector_list