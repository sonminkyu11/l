from re import X
import numpy as np
from math import *
from scipy import interpolate as ip
import sys
import os
import socket
import matplotlib.pyplot as plt


sys.path.append(os.getcwd())
hostname = socket.gethostname()

HOST_NAME = '/home/' + hostname

class LaneChange():
    def __init__(self):
        self.ego_vy = 0
        self.temp_ego_vy = 0
        self.ego_ay = 0
        self.ego_vy_limit = 50/3.6

        self.front_limit_dis = 5
        self.rear_limit_dis = 4

        self.lidar_front = self.lidar_left = self.lidar_right = []
        self.temp_left = self.temp_right = []
        self.able_left = self.able_right = 0
        
        self.temp_front = np.array([])
        self.target_front = self.target_temp_front = np.array([])
        self.target_rear = self.target_temp_rear = np.array([])

        self.map = 1
        self.np_size = 0

        self.global_path_lc = self.obs_lc = self.ovt_lc = 0
        self.lc_direction = self.lc_step = self.lc_direction_final = 0
        self.lc_velocity = 99
        self.lc_ing = 0
        self.path_idx = False
        self.reset = 0

        self.data_changed = False
        self.is_lane = False
        self.is_obstacle = False

        self.front_space_safe = self.rear_space_safe = False
        self.front_safe_space = self.rear_safe_space = self.front_sdi = self.rear_sdi = False
        self.tau_abs = 1.5
        self.tau_rel = 0.8
        self.minimum_distance = 5


        self.str_light = self.str_light_remain = 0
        self.left_light = self.left_light_remain = 0
        self.ped1_light = self.ped1_light_remain = 0
        self.ped2_light = self.ped2_light_remain = 0

        self.current_id = 0
        self.version = 0
        self.path_id = 0

        self.remain_distance = 0

        self.ego_x = self.ego_y = self.ego_h = 0
        self.local_path_x = self.local_path_y = self.local_path_s = []


        self.avoid_count = 0
        self.lc_ing_count = 0
        self.control_status = 0
        self.green_light = 0
        # self.nearest_fr = self.get_nearest_fr()

        self.obs_rel_bwn = ip.interp1d( [-5,-1,0,5/3.6,10/3.6,15/3.6,20/3.6,25/3.6,30/3.6,35/3.6,40/3.6,45/3.6,50/3.6, 55/3.6, 60/3.6, 65/3.6, 70/3.6, 75/3.6, 80/3.6, 85/3.6, 90/3.6],\
            [2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.7, 2.9, 3.1, 3.3, 3.5, 3.7, 3.9, 4.1, 4.3, 4.5] )
        self.lc_rel_bwn_f = ip.interp1d( [-5,-1,0,5/3.6,10/3.6,15/3.6,20/3.6,25/3.6,30/3.6,35/3.6,40/3.6,45/3.6,50/3.6, 55/3.6, 60/3.6, 65/3.6, 70/3.6, 75/3.6, 80/3.6, 85/3.6, 90/3.6],\
            [10,10,10,15,18,21,24,27,30,33,36,39,42, 45, 48, 51, 54, 57, 60, 63, 66])
        self.lc_rel_bwn_r = ip.interp1d( [-5,-1,0,5/3.6,10/3.6,15/3.6,20/3.6,25/3.6,30/3.6,35/3.6,40/3.6,45/3.6,50/3.6,55/3.6, 60/3.6, 65/3.6, 70/3.6, 75/3.6, 80/3.6, 85/3.6, 90/3.6],\
            [-5,-5,-5,-6,-8,-10,-12,-14,-14,-12,-10,-8,-6, -6, -6, -6, -6, -7, -8, -9,-10])

        self.extended_lane = [10, 26, 33, 48, 67, 85, 98]
        self.exception_lane = [11, 32, 99, 86, 68, 49, 45, 59, 54]

    def select_map(self, map_number):
        map_name = 'dmc_map'

        if map_number == 1:   # Sangam DMC
            map_name = 'dmc_map'
        elif map_number == 2: # K-City (center)
            map_name = 'k_city_map'
        elif map_number == 3: # K-City (highway)
            map_name = 'pg_map'
        elif map_number == 4: # CarMaker DMC
            map_name = 'carmaker_dmc_map'
        elif map_number == 5: # K-City (slope, left)
            map_name = 'slope_left_map'
        elif map_number == 6: # K-City (slope, right)
            map_name = 'slope_right_map'
        elif map_number == 7: # CarMaker K-City (highway)
            map_name = 'carmaker_pg_map'
        
        return map_name

    def get_path_distance(self, path_id):
        # map_name = self.select_map(map_number)
        map_name = 'dmc_map'
        roads = HOST_NAME + '/save_ws/maps/'+ map_name + \
            '/Roads/M_Road_{}.txt'
        path_info = np.loadtxt(roads.format(path_id), dtype=np.float64)
        path_distance = path_info[-1, 2]

        return path_distance

    def get_distance_from_map(self, lidar_data):
        # if np.size(self.local_path_x) >= 1:
        idx_obj = np.argmin(np.square(self.local_path_x - lidar_data[1]) + \
            np.square(self.local_path_y - lidar_data[2]))
        idx_ego = np.argmin(np.square(self.local_path_x) + np.square(self.local_path_y))
        if lidar_data[2] > 0:
            distance = self.local_path_s[idx_obj] - self.local_path_s[idx_ego] - 2.6 -2
        else:
            distance = self.local_path_s[idx_obj] - self.local_path_s[idx_ego] + 2 + 2.6
        
        return abs(distance)
    
    def get_path_cumsum(self, lc_path):
        local_s = np.sqrt((lc_path[0][1:] - lc_path[0][:-1])**2 + \
            (lc_path[1][1:] - lc_path[:-1])**2)
        local_cumsum = np.cumsum(local_s)

        return local_cumsum

    def get_eta(self, lidar_data, ego=True, front=True):
        if ego:
            return self.remain_distance / (self.ego_vy + 0.001)

        else:
            distance = self.get_distance_from_map(lidar_data)
            if front:
                return (self.remain_distance - distance - 4) / (lidar_data[-1] + self.ego_vy + 0.001)
            else:
                return (self.remain_distance + distance) / (lidar_data[-1] + self.ego_vy + 0.001)

    def is_front_rear(self):
        front = rear = np.array([])
        if self.lc_direction == -1:
            lidar_data = self.lidar_left

        elif self.lc_direction == 1:
            lidar_data = self.lidar_right

        if np.shape(lidar_data)[0] == 1:
            if lidar_data[0,2] > 0:
                front = lidar_data[0]
            else:
                rear = lidar_data[0]

        elif np.shape(lidar_data)[0] >= 2:
            for i in range(len(lidar_data)):
                temp = lidar_data[i]
                if lidar_data[i,2] > 0:
                    if np.size(front) == 0:
                        front = temp
                    else:
                        front = np.vstack((front, temp))
                else:
                    if np.size(rear) == 0:
                        rear = temp
                    else:
                        rear = np.vstack((rear, temp))

        if np.size(front) >= 1:
            self.target_front = self.get_nearest_obj(front)
        else:
            self.target_front  = front
        
        if np.size(rear) >= 1:
            self.target_rear = self.get_nearest_obj(rear)
        else:
            self.target_rear = rear

    def get_nearest_obj(self, lidar_data):
        if np.size(lidar_data) > self.np_size + 1:
            id = self.get_nearest_id(lidar_data)
            row = np.where(lidar_data == id)[0]
            lidar_data = lidar_data[row][0]

        return lidar_data

    def get_absv(self, lidar_data):
        return lidar_data[-1] + self.ego_vy

    def check_ovt(self):
        self.ovt_lc = 0
        if np.size(self.lidar_front) >= 1 and self.ego_vy >= 5 and self.remain_distance != 9999.0:
            distance = self.get_distance_from_map(self.lidar_front[0])
            safe_distance = max(self.tau_abs * self.ego_vy + self.tau_rel * - self.lidar_front[0][-1], 5)
            front_eta = self.get_eta(self.lidar_front[0],ego=False)
            v_limit_eta = (self.remain_distance - distance - 4) / self.ego_vy_limit

            if (front_eta - v_limit_eta) * (self.lidar_front[0][-1] + self.ego_vy) > safe_distance + self.ego_vy_limit * 4 and distance < 35:
                if self.able_right >= 1 and np.size(self.lidar_right) == 0:
                    self.ovt_lc = 1

                elif self.able_left >= 1 and np.size(self.lidar_left) == 0:
                    self.ovt_lc = -1

        else:
            pass

    def get_nearest_id(self, lidar_data):
        row = np.argmin(abs(lidar_data[:,2]))
        id = lidar_data[row,0]

        return id

    def get_array_size(self):
        if self.version == 3:
            np_size = 7
        else:
            np_size = 12

        return np_size

    def able_lc_direction(self):
        self.lc_step = 0
        self.front_space_safe = self.rear_space_safe = False

        # 0
        self.is_front_rear()        

        # print('target front: ', self.target_front)
        # print('target rear: ', self.target_rear)

        # 1st
        self.check_lane()
        # print('check there is lane: ', self.is_lane)
        
        # 2nd
        if self.lc_step == 1:
            extended_lane = self.extended_lane.count(self.path_id)
            if self.lc_direction == 1 and self.path_id == 98:
                self.lc_step = 3
            
            elif extended_lane == 1:
                self.lc_step = 3

            else:
                self.check_obstacle(front=False)
        # print('check there is obastacle: ', self.is_obstacle)

        # 3rd
        if self.lc_step == 2:
            self.check_space()
        
        if self.lc_step == 2.5:
            self.determine_vel()
            # if self.global_path_lc == 0:
            #     self.determine_vel()

        if self.lc_step == 3:
            self.lc_direction_final = self.lc_direction
        
        else:
            self.lc_direction_final = 0

        print('lc final direction: ', self.lc_direction_final)
        print('check lc step: ', self.lc_step)

    def check_data_changed(self):
        self.data_changed = False
        if np.size(self.target_front) >= 1 and np.size(self.target_temp_rear) >= 1:
            if self.target_front[0] == self.target_temp_rear[0]:
                self.data_changed = True

        if np.size(self.target_rear) >= 1 and np.size(self.target_temp_rear) >=1:
            if self.target_rear[0] == self.target_temp_rear[0]:
                self.data_changed = True
        
        if self.data_changed:
            self.lc_step = 0
        # return data_changed

    def check_lane(self):
        if self.lc_direction == -1 and self.able_left >= 1:
            self.is_lane = True
        
        elif self.lc_direction == 1 and self.able_right >= 1:
            self.is_lane = True
        
        if self.is_lane:
            self.lc_step = 1

    def check_obstacle(self, front=True):
        obstacle = False
        obs_check = []
        if front:
            if np.size(self.lidar_front) >= 1 and np.size(self.temp_front) >= 1:
                lidar_data = self.lidar_front[0]

            else:
                lidar_data = []
                temp_data = []
        else:
            lidar_data = self.target_front
            temp_data = self.target_temp_front

        # print('lidar front: ', lidar_data)
        if np.size(lidar_data) >= 1:
            # print('obstacle front data: ', lidar_data)
            abs_v = self.get_absv(lidar_data)
            distance = self.get_distance_from_map(lidar_data)
            x_relvel = lidar_data[-2]
            # obs_check = np.where( (abs_v <= self.obs_rel_bwn(self.ego_vy)) & (self.ego_vy >= 2))[0]

            if front:
                # print('front')
                # print('abs v: ', abs_v)
                # print('lidar data: ', lidar_data[2])
                # print('distance: ', distance)
                # print('x_relvel: ', x_relvel)
                if distance < 35 and (self.ego_vy >= 4.5 or self.remain_distance > 75):
                    obs_check = np.where( abs_v <= self.obs_rel_bwn(self.ego_vy))[0]

            else:
                obs_check = np.where( abs_v <= self.obs_rel_bwn(self.ego_vy))[0]

            if np.size(obs_check) >= 1:
                obstacle = True

        self.is_obstacle = obstacle

        if front:
            if obstacle:
                obstacle = 1
            else:
                obstacle = 0

            self.obs_lc = obstacle

        else:
            if not self.is_obstacle:
                self.lc_step = 2

    def check_space(self):
        safe = False
        if np.size(self.target_front) >= 1:
            self.front_safe_space = max(self.tau_abs * self.ego_vy + self.tau_rel * - self.target_front[-1], 3)
            front_distance = self.get_distance_from_map(self.target_front)

            # print('front safe distance: ', self.front_safe_space)
            # print('front distance:', front_distance)
            if self.front_safe_space < front_distance:
                self.front_space_safe = True
            
        else:
            self.front_space_safe = True
            
        if np.size(self.target_rear) >= 1:
            self.rear_safe_space = max(self.tau_abs * (self.target_rear[-1] + self.ego_vy) + self.tau_rel * self.target_rear[-1], 2)
            rear_distance = self.get_distance_from_map(self.target_rear)

            # print('rear safe distance: ', self.rear_safe_space)
            # print('rear distnace: ', rear_distance)
            if self.rear_safe_space < rear_distance:
                self.rear_space_safe = True

        else:
            self.rear_space_safe = True

        if self.front_space_safe and self.rear_space_safe:
            safe = True

        if safe:
            self.lc_step = 3

        else:
            self.lc_step = 2.5

        # print('final safe: ', safe)

    def determine_vel(self):
        ego_eta = self.get_eta([], ego=True)
        # print("ego_eta :", ego_eta)
        front_eta = rear_eta = vel_fr = vel_rr = 0

        if np.size(self.target_front) >= 1:
            front_eta = self.get_eta(self.target_front, ego=False)
            # print("front_eta :", front_eta)
        
        if np.size(self.target_rear) >= 1:
            rear_eta = self.get_eta(self.target_rear, ego=False, front=False)
            # print("rear_eta :", rear_eta)

        if front_eta != 0:
            front_absv = self.get_absv(self.target_front)
            front_safe_distance = max(self.tau_abs * (self.target_front[-1] + self.ego_vy) + self.tau_rel * - self.target_front[-1], 5)
            # print('self.target_front : ',self.target_front[-1])
            # print('remain distance: ', self.remain_distance)
            # print('possible distance: ', (front_eta - ego_eta) * front_absv)
            # print("safe distance : ",front_safe_distance + self.ego_vy * 4)
            if (front_eta - ego_eta) * front_absv > front_safe_distance + self.ego_vy * 4:
                vel_fr = 0
            else:
                vel_fr = -1

        else:
            vel_fr = 0

        if rear_eta != 0:
            rear_absv = self.get_absv(self.target_rear)
            rear_safe_distance = max(self.tau_abs * (self.target_rear[-1] + self.ego_vy) + self.tau_rel * self.target_rear[-1], 5)
            # print('self.target_rear : ',self.target_rear[-1])
            # print("rs distance : ",rear_safe_distance)
            # print('remain distance: ', self.remain_distance)
            # print('possible distance: ', (rear_eta - ego_eta) * rear_absv)
            # print("safe distance : ",rear_safe_distance + self.ego_vy * 4)
            if (rear_eta - ego_eta) * rear_absv > rear_safe_distance + self.ego_vy * 4:
                vel_rr = 0
            else:
                vel_rr = -1
        
        else:
            vel_rr = 0

        vel = min(vel_fr, vel_rr)
        # print("final vel : ",vel)

        cum_distance = self.get_path_distance(self.path_id)

        if vel == -1:
            # print('eta: ', ego_eta)
            self.lc_velocity = max( self.ego_vy - (1 - self.remain_distance/cum_distance) * 30/3.6 ) * 3.6, 10)

        else:
            self.lc_velocity = 99

        print('cum distance: ', cum_distance)
        print('target vel: ', self.lc_velocity)

    def determine_status(self, lidar_data, temp):
        acc = 'no object'

        if np.size(lidar_data) >= 1 and np.size(temp) >= 1:
            current_vel = lidar_data[-1] + self.ego_vy
            past_vel = temp[-1] + self.ego_vy

            data_accel = (current_vel - past_vel) / 0.1

            if data_accel >= 0.15:
                acc = 'accel'
            elif data_accel <= -0.15:
                acc = 'decel'
            elif data_accel < 0.15 and data_accel > - 0.15 and (lidar_data[2] + self.ego_vy) >= 2:
                acc = 'uniform'
            else:
                acc = 'idling'

        return acc

    def get_ssd(self, lidar_data, ego=False):
        reaction_time = 0.8
        friction_coeff = 0.8
        if ego:
            vel = self.ego_vy * 3.6 #[kph]
        else:
            vel = self.get_absv(lidar_data) * 3.6 #[kph]

        ssd = (vel ** 2 / 254 * friction_coeff) + \
            reaction_time * vel * 0.278
        return ssd
    
    def get_sdi(self, lidar_data, distance, front=True):
        safe = True
        if np.size(lidar_data) >= 1:
            object_length = 4 #temp
            object_ssd = self.get_ssd(lidar_data, ego=False)
            ego_ssd = self.get_ssd(lidar_data, ego=True)

            if front:
                sdi = distance + object_ssd - ego_ssd
            else:
                sdi = distance + ego_ssd - object_ssd

        return sdi

    def determine_avoiding(self):
        # candidate = [self.str_light, self.left_light, self.ped1_light, self.ped2_light]
        # remain_time_candidate = [self.str_light_remain, self.left_light_remain, self.ped1_light_remain, self.ped2_light]

        # if self.green_light == 1:
        #     green_idx = candidate.index(1)
        #     remain_time = remain_time_candidate[green_idx]

        if np.size(self.lidar_front) >= 1:
            abs_v = self.get_absv(self.lidar_front[0])
            distance = self.get_distance_from_map(self.lidar_front[0])
            obs_check = np.where( (abs_v <= self.obs_rel_bwn(self.ego_vy)))[0]

            if self.control_status == 6:
                if self.ego_vy >= -0.5 and self.ego_vy <= 0.5:
                    if np.size(obs_check) >= 1 and distance < 10:
                        if self.lc_ing != 0:
                            self.lc_ing_count += 1
                        
                        elif self.path_id > 1000 or self.green_light == 1:
                            self.avoid_count += 1
                        
            
                    #     if self.path_id > 1000:
                    #         self.avoid_count
        

                    # if self.path_id > 1000:
                    #     if np.size(obs_check) >= 1 and distance < 10:
                    #         self.avoid_count += 1
                    
                    # else:
                    #     if self.green_light == 1:
                    #         if np.size(obs_check) >= 1 and distance < 10:
                    #             self.avoid_count += 1
                
                else:
                    self.avoid_count = 0

            weight = max(round(self.remain_distance / 7),1)
            if weight >= 2:
                weight = 2
            # print('weight: ', weight)

            if self.lc_ing != 0:
                if self.lc_ing_count >= 40 * weight:
                    self.reset = 1
                    self.lc_ing_count = 0
            
            elif self.avoid_count >= 40 * weight:
                self.obs_lc = 1

            # if self.avoid_count >= 40 * weight:
            #     if self.lc_ing != 0:
            #         self.reset = 1
            #         self.lc_ing_count = 0
                
            #     else:
            #         self.obs_lc = 1

            if self.lc_direction_final != 0:
                self.avoid_count = 0

    def set_lc_distance(self):
        if self.map == 3:
            distance = float(85)

        else:
            if self.obs_lc == 1:
            # np.size(self.lidar_front) >= 1:
                if self.lidar_front[0][-1] > 0:
                    distance = max(float(self.get_distance_from_map(self.lidar_front[0])) + (self.ego_vy + self.lidar_front[0][-1]) * 3, 15)
                else:
                    distance = max(float(self.get_distance_from_map(self.lidar_front[0])), 15)

            else:
                if np.size(self.target_front) >= 1:
                    if self.target_front[-1] > 0:
                        distance = self.get_distance_from_map(self.target_front) + (self.ego_vy + self.target_front[-1])*2
                    else:
                        distance = self.get_distance_from_map(self.target_front) + (self.ego_vy + self.target_front[-1]) *3
                    
                else:
                   distance = max(self.ego_vy * 5, 30.0)
            
            if self.remain_distance < distance:
                distance = self.remain_distance

        # print('bezier distance: ', distance)

        return distance, max(distance-10,15), max(distance-15,15)

    def get_lc_path_idx(self,path1, path2, path3):
        idx = 0
        safe = True
        if self.lc_direction_final == -1:
            path1[0] += 0.9025
            path1[1] += 3.575
            path2[0] += 0.9025
            path2[1] += 3.575
            path3[0] += 0.9025
            path3[1] += 3.575
        
        else:
            path1[0] -= 0.9025
            path1[1] += 3.575
            path2[0] -= 0.9025
            path2[1] += 3.575
            path3[0] -= 0.9025
            path3[1] += 3.575

        if np.size(self.lidar_front) >= 1:
            lidar_front = self.lidar_front[0]

            if self.lc_direction_final == -1:
                edge_point_x = lidar_front[1] - lidar_front[3] / 2
                edge_point_y = lidar_front[2] - lidar_front[4] / 2

            else:
                edge_point_x = lidar_front[1] + lidar_front[3] / 2
                edge_point_y = lidar_front[2] - lidar_front[4] / 2

            path1_min_dis = np.min(np.sqrt(np.square(path1[0] - edge_point_x) + \
                np.square(path1[1] - edge_point_y)))
            path2_min_dis = np.min(np.sqrt(np.square(path2[0] - edge_point_x) + \
                np.square(path2[1] - edge_point_y)))
            path3_min_dis = np.min(np.sqrt(np.square(path3[0] - edge_point_x) + \
                np.square(path3[1] - edge_point_y)))

            if path1_min_dis > 1:
                idx = 0
            elif path2_min_dis > 1:
                idx = 1
            else:
                idx = 2

            # else:
            #     safe = False

        return idx

    def set_lc_direction(self, global_path_lc, obs_lc, ovt_lc, forced_lc):
        direction = 0

        if obs_lc:
            if self.able_right >= 1:
                if np.size(self.lidar_right) == 0:
                    obs_lc = 1

                elif self.able_left >= 1:
                    if np.size(self.lidar_left) == 0:
                        obs_lc = -1

                else:
                    obs_lc = 1

            else:
                obs_lc = -1

        candidate = np.array([global_path_lc, obs_lc, ovt_lc, forced_lc])
        idx = np.nonzero(candidate)[0]

        # print('candidate: ', candidate)
        # print('direction idx: ', idx)
        # print('idx size: ', np.size(idx))
        if np.size(idx) >= 2:
            direction = candidate[0]
            
        elif np.size(idx) == 1:
            direction = candidate[idx[0]]
        
        else:
            direction = 0

        self.lc_direction = direction
        # print('lc direction: ', self.lc_direction)

    def update(self, temp_front, temp_target_front, temp_target_rear, version, ego_vy, control_status):
        # self.nearest_fr = self.get_nearest_fr()
        self.np_size = self.get_array_size()
        self.temp_front = temp_front
        self.target_temp_front = temp_target_front
        self.target_temp_rear = temp_target_rear
        self.version = version
        self.temp_ego_vy = ego_vy
        self.control_status = control_status