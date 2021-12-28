from math import dist
from re import L
import rclpy
from rclpy.node import Node
from std_msgs.msg import *
# from rclpy.qos import QoSProfile
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
from custom_msgs.msg import *
from custom_msgs.srv import *
import sys
import numpy as np
from .lane_change import LaneChange
from .IDM import IDM
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# from lane_change import checking_path

class DecisionMaking(Node):
    def __init__(self):
        super().__init__('dm')
        timer_period = 0.1
        self.qos_profile = QoSProfile(depth=1)

        self.future = None
        self.version_init = False

        self.forced_lc = 0
        self.service_received = False

        self.lc_ing = 0
        self.lc_status = False
        self.traffic_light = np.zeros(8)

        self.go_triger = 1
        self.green_light = 1
        self.control_status = 0
        self.v2x_dm = False
        self.brk_pressure = 0

        self.waypoint_reset = 0

        self.lc_distance = 0
        self.distance1 = self.distance2 = self.distance3 = 0
        self.count = 0
        self.lap = 0
        self.crazy = False
        
        self.lc = LaneChange()

        #Subscription
        self.forced_lc_subscriber = self.create_subscription(
            Int16, '/forced_lc', self.forced_lc_callback, self.qos_profile)

        self.map_subscriber = self.create_subscription(
            Int16, '/map', self.map_callback, self.qos_profile)

        self.version_subscriber = self.create_subscription(
            Int16, '/version', self.version_callback, self.qos_profile)

        self.localization_subscriber = self.create_subscription(
            Float64MultiArray, 'ego_location', self.localization_callback, self.qos_profile)

        self.local_path_subscriber = self.create_subscription(
                    Paths, 'local_path', self.local_path_callback, self.qos_profile)
        
        self.possible_lanes_subscriber = self.create_subscription(
            AroundPaths, 'around_paths', self.able_lanes_callback, self.qos_profile)
        
        self.waypoint_reset_sub = self.create_subscription(
            Int16, 'waypoint_reset', self.waypoint_reset_callback, self.qos_profile)

        self.lc_request_subscriber = self.create_subscription(
            Int16, '/lc_request', self.global_lc_callback, self.qos_profile)
        
        self.lc_ing_subscriber = self.create_subscription(
                    Int16, 'lc_ing', self.lc_ing_callback, self.qos_profile)
        
        self.traffic_light_subscriber = self.create_subscription(
                    NiroCAN1568, 'CAN/id1568', self.traffic_light_callback, self.qos_profile)

        self.path_id_subscriber = self.create_subscription(
                    Int16, 'path_id', self.path_id_callback, self.qos_profile)

        self.v2x_go_triger_subscriber = self.create_subscription(
                    Float64MultiArray, 'steer_status', self.v2x_go_triger_callback, self.qos_profile)

        self.control_status_subscriber = self.create_subscription(
                    Float64MultiArray, 'control_rqt', self.control_status_callback, self.qos_profile)

        self.create_subscription(NiroCAN304(), 'CAN/id304', self.brk_pressure_callback, self.qos_profile)

        self.create_subscription(
                    Int16, 'crazy_mode', self.crazy_mode_callback, self.qos_profile)

        #Publish
        self.lc_path_idx_publisher = self.create_publisher(
            Int16, '/lc_path_idx', self.qos_profile)

        self.lc_velocity_publisher = self.create_publisher(
            Float32, '/lc_velocity', self.qos_profile)

        self.final_lc_direction_publisher = self.create_publisher(
            Int16, '/final_lc_direction', self.qos_profile)

        self.path_reset_publisher = self.create_publisher(
            Int16, '/path_reset', self.qos_profile)
        
        self.lc_distance_publisher = self.create_publisher(
            Float32, '/lc_distance', self.qos_profile)

        #Timer initializtion
        self.timer = self.create_timer(timer_period, self.timer_callback)

        #Service client
        self.client = self.create_client(GetBezier, 'get_bezier')
        self.req = GetBezier.Request()

    #Subscriber callback declaration
    def forced_lc_callback(self, msg):
        if self.forced_lc == 0:
            self.forced_lc = msg.data

    def map_callback(self, msg):
        # 1 = sangam(real) / 2 = k-city / 3 = k-city(highway) / 4 = sangam(carmaker)
        self.lc.map = msg.data

    def version_callback(self, msg):
        # 1 = avante / 2 = niro / 3 = carmaker
        self.lc.version = msg.data

        # lc.overtaking_dm(msg)
        if self.version_init == False:
            if self.lc.version == 1:
                self.vehicle = 'Avante'
                self.ego_velocity_subscriber = self.create_subscription(
                    SaveCAN1810(), 'CAN/id1810', self.ego_vel_callback, self.qos_profile)
                self.ego_acc_subscriber = self.create_subscription(
                    SaveCAN1811(), 'CAN/id1811', self.ego_acc_callback, self.qos_profile)
                self.lidar_front_subscriber = self.create_subscription(
                    Float64MultiArray, 'lidar_data_front', self.lidar_front_callback, self.qos_profile)
                self.lidar_left_subscriber = self.create_subscription(
                    Float64MultiArray, 'lidar_data_left', self.lidar_left_callback, self.qos_profile)
                self.lidar_right_subscriber = self.create_subscription(
                    Float64MultiArray, 'lidar_data_right', self.lidar_right_callback, self.qos_profile)

            elif self.lc.version == 2:
                self.vehicle = 'Niro'
                self.ego_velocity_subscriber = self.create_subscription(
                    NiroCAN640(), 'CAN/id640', self.ego_vel_callback, self.qos_profile)
                
                self.lidar_front_subscriber = self.create_subscription(
                    Float32MultiArray, 'lidar_data_front', self.lidar_front_callback, self.qos_profile)
                self.lidar_left_subscriber = self.create_subscription(
                    Float32MultiArray, 'lidar_data_left', self.lidar_left_callback, self.qos_profile)
                self.lidar_right_subscriber = self.create_subscription(
                    Float32MultiArray, 'lidar_data_right', self.lidar_right_callback, self.qos_profile)
                
            elif self.lc.version == 3:
                self.vehicle = 'CM'
                self.ego_velocity_subscriber = self.create_subscription(
                    Float64MultiArray, 'vehicle_info_out2', self.ego_vel_callback, self.qos_profile)
                self.lidar_subscriber = self.create_subscription(
                    Float64MultiArray, 'objectinfo_out2', self.lidar_obj_sensor_callback, self.qos_profile)

            self.version_init = True

    def ego_vel_callback(self, msg):
        if self.lc.version == 1:    # Avante
            self.lc.ego_vy = ((msg.wheel_spd_rl + msg.wheel_spd_rr) / 2) / 3.6 # [kph] -> [m/s]

        elif self.lc.version == 2:  # Niro
            self.lc.ego_vy = ((msg.gway_wheel_velocity_rl + msg.gway_wheel_velocity_rr) / 2) / 3.6 # [kph] -> [m/s]
            
        elif self.lc.version == 3:  # CM
            self.lc.ego_vy = msg.data[3] #[m/s]
            self.lc.ego_ay = msg.data[9]

    def localization_callback(self, msg):
        self.lc.ego_x = msg.data[0]
        self.lc.ego_y = msg.data[1]
        self.lc.ego_h = msg.data[2]
        self.lc.current_id = msg.data[3]
        self.lc.remain_distance = msg.data[-1]

    def local_path_callback(self, msg):
        self.lc.local_path_x = np.array(msg.x)
        self.lc.local_path_y = np.array(msg.y)
        self.lc.local_path_s = np.array(msg.s)

    def path_id_callback(self, msg):
        self.lc.path_id = msg.data
        
    def able_lanes_callback(self, msg):
        self.lc.able_left = msg.left_lane
        self.lc.able_right = msg.right_lane

        no_left = [9, 16, 85, 26, 37, 91, 47, 53, 95, 67, 10, 97]
        no_right = [2, 27, 33, 34, 54, 73, 111, 92, 79]

        left_count = no_left.count(self.lc.path_id)
        right_count = no_right.count(self.lc.path_id)

        if self.waypoint_reset == 1:
            if self.lc.path_id == 48:
                self.lc.able_right = 0
                self.lc.able_left = 0
            
            if self.lc.path_id == 1 or self.lc.path_id == 2 or self.lc.path_id == 3:
                self.waypoint_reset = 0

        # else:
        #     if self.lc.path_id == 48:
        #         self.lc.able_left = 0

        if left_count == 1:
            self.lc.able_left = 0
        
        if right_count == 1:
            self.lc.able_right = 0

    def lidar_front_callback(self, msg):
        self.lc.lidar_front = np.array(msg.data)
        self.lc.lidar_front = self.lc.lidar_front.reshape(-1, 12)
        trash = np.where( (self.lc.lidar_front[:, 0] == 9999.0) | \
            (self.lc.lidar_front[:,5] - self.lc.lidar_front[:,6] > 2) | (self.lc.lidar_front[:,2] < 0))[0]
        self.lc.lidar_front = np.delete(self.lc.lidar_front, trash, axis=0)
        # print('front: ', self.lc.lidar_front)

        # id, x, y, w, h, z, minz,relax, relay, class, relvx, relvy
        
    def lidar_left_callback(self, msg):
        self.lc.lidar_left = np.array(msg.data)
        self.lc.lidar_left = self.lc.lidar_left.reshape(-1, 12)
        trash = np.where( (self.lc.lidar_left[:, 0] == 9999.0) | \
            (self.lc.lidar_left[:,5] - self.lc.lidar_left[:,6] > 2))[0]
        self.lc.lidar_left = np.delete(self.lc.lidar_left, trash, axis=0)

    def lidar_right_callback(self, msg):
        self.lc.lidar_right = np.array(msg.data)
        self.lc.lidar_right = self.lc.lidar_right.reshape(-1, 12)
        # trash = np.where( (self.lc.lidar_right[:, 0] == 9999.0) | \
        #     (self.lc.lidar_right[:,5] - self.lc.lidar_right[:,6] > 2))[0]
        trash = np.where( self.lc.lidar_right[:, 0] == 9999.0)[0]
        self.lc.lidar_right = np.delete(self.lc.lidar_right, trash, axis=0)

        # print('right data: ', self.lc.lidar_right)

    def lidar_obj_sensor_callback(self, msg):
        # obj_num, id, x, y, w, h, vx, vy
        object_sensor = np.array(msg.data)

        if object_sensor[0] == 0:
            self.lc.lidar_front = []
            self.lc.lidar_left = []
            self.lc.lidar_right = []

        else:
            object_sensor = object_sensor[1:].reshape(-1,7)
            object_sensor = object_sensor * 1.2 # noise
            front_idx = np.where( (object_sensor[:,1] <= 1.5) & (object_sensor[:,1] >= -1.5) & (object_sensor[:,2] > 0) & (object_sensor[:,2] <= 40))[0]
            left_idx = np.where( (object_sensor[:,1] < -1.5) & (object_sensor[:,1] > -4.5) & (object_sensor[:,2] <= 40) & (object_sensor[:,2] > -20) )[0]
            right_idx = np.where((object_sensor[:,1] > 1.5) & (object_sensor[:,1] < 4.5) & (object_sensor[:,2] <= 40) & (object_sensor[:,2] > -20) )[0]

            self.lc.lidar_front = object_sensor[front_idx, :]
            self.lc.lidar_left = object_sensor[left_idx, :]
            self.lc.lidar_right = object_sensor[right_idx, :]

            # print('lidar front: ', self.lc.lidar_front)

    def global_lc_callback(self, msg):
        if self.crazy == True:
            if self.lc.path_id == 37 or self.lc.path_id == 96 or self.lc.path_id == 95:
                self.lc.global_path_lc = 1
            else:
                if np.size(self.lc.lidar_front) >= 1:
                    if self.lc.able_left >= 1:
                        self.lc.global_path_lc = -1 
                    elif self.lc.able_right >= 1:
                        self.lc.global_path_lc = 1
                else:
                    self.lc.global_path_lc = 0

        else:
            self.lc.global_path_lc = msg.data
            
    def lc_ing_callback(self, msg):
        self.lc_ing = msg.data
        self.lc.lc_ing = msg.data

        if self.lc_ing == 1 or self.lc_ing == -1:
            self.lc_status = True
            self.lc.lc_direction_final = 0
            self.lc.lc_velocity = 99

        else:
            self.lc_status = False
            
    def traffic_light_callback(self, msg):
        '''
        status -> uint16 / remain time -> float32
        status -> 0: red / 1: green / 2: yellow, blinking green at ped / 3: no signal
        ped1 -> front pedestrian light
        ped2 -> light pedestrain light
        '''

        if self.lc.version == 2: # niro
            self.lc.str_light = msg.str_status
            self.lc.str_light_remain = msg.str_time_remain
            self.lc.left_light = msg.left_status
            self.lc.left_light_remain = msg.left_time_remain
            self.lc.ped1_light = msg.ped1_status
            self.lc.ped1_light_remain = msg.ped1_time_remain
            self.lc.ped2_light = msg.ped2_status
            self.lc.ped2_light_remain = msg.ped2_time_remain

        else:
            self.lc.str_light = self.lc.left_light = self.lc.ped1_light = self.lc.ped2_light = 3
            self.lc.str_light_remain = self.lc.left_light_remain = self.lc.ped1_light_remain = self.lc.ped2_light_remain = 9999

    def v2x_go_triger_callback(self, msg):
        self.go_triger = int(msg.data[-2])
        self.green_light = int(msg.data[-1])
        self.lc.green_light = int(msg.data[-1])

    def control_status_callback(self, msg):
        self.control_status = int(msg.data[-2])
    
    def brk_pressure_callback(self, msg):
        self.brk_pressure = msg.gway_brakemastercylinderpressure
        
    def waypoint_reset_callback(self, msg):
        if self.waypoint_reset == 0:
            self.waypoint_reset = msg.data

    def crazy_mode_callback(self, msg):
        crazy_mode = msg.data
        if crazy_mode == 1:
            self.crazy = True

        
    #Publish function declaration
    def publish_lc_path_idx(self, path_idx):
        print("pub lc idx", path_idx)
        self.inform_status()
        msg = Int16()
        msg.data = path_idx
        self.lc_path_idx_publisher.publish(msg)
        self.lc.global_path_lc = self.lc.obs_lc = self.lc.ovt_lc = self.forced_lc = 0
        self.lc.is_lane = self.lc.obstacle = self.lc.is_obstacle = False
        self.lc_status = True

    def publish_lc_velocity(self, vel):
        msg = Float32()
        msg.data = float(vel)
        self.lc_velocity_publisher.publish(msg)

    def publish_path_reset(self):
        msg = Int16()
        msg.data = self.lc.reset
        self.path_reset_publisher.publish(msg)
        self.lc.reset = 0
        self.lc.obs_lc = self.lc.ovt_lc = self.forced_lc = 0
        self.lc.lc_direction = self.lc.lc_direction_final = 0
    
    def publish_lc_distance(self, distance):
        msg = Float32()
        msg.data = float(distance)
        self.lc_distance_publisher.publish(msg)

    def send_request(self):
        print("service send")
        self.distance1, self.distance2, self.distance3 = self.lc.set_lc_distance()
        self.req.direction = int(self.lc.lc_direction_final)
        self.req.distance1 = float(self.distance1)
        self.req.distance2 = float(self.distance2)
        self.req.distance3 = float(self.distance3)
        self.future = self.client.call_async(self.req)

    def request_lc_path(self): 
        if not self.service_received == True:
            if self.lc.lc_direction_final != 0:
                self.send_request()
                # self.forced_lc = 0
                self.service_received = True

        if not self.future == None: 
            if self.future.done(): 
                try: 
                    response = self.future.result() 
                    path1_x = response.path1_x 
                    path1_y = response.path1_y 
                    path2_x = response.path2_x 
                    path2_y = response.path2_y 
                    path3_x = response.path3_x 
                    path3_y = response.path3_y 

                    possible_lc_path1 = np.array([path1_x, path1_y])
                    possible_lc_path2 = np.array([path2_x, path2_y])
                    possible_lc_path3 = np.array([path3_x, path3_y])

                    idx = self.lc.get_lc_path_idx(possible_lc_path1, possible_lc_path2, possible_lc_path3) 
                    # print('idx: ', idx) 
                    # plt.figure(figsize=(20/2.54, 60/2.54)) 
                    # plt.axis('equal') 
                    # plt.grid() 
                    # plt.plot(path1_x, path1_y, 'ro') 
                    # plt.plot(path2_x, path2_y, 'ko') 
                    # plt.plot(path3_x, path3_y, 'bo') 
                    # plt.show() 
                    # plt.clf() 
                    self.future = None 
                    # print("future done") 

                    if idx == 1:
                        self.lc_distance = float(self.distance1)
                    elif idx == 2:
                        self.lc_distance = float(self.distance2)
                    else:
                        self.lc_distance = float(self.distance3)
                    

                    self.publish_lc_path_idx(idx)
                    
                    self.front_array = self.rear_array = [] 
                    self.service_received = False 

                except Exception as e: 
                    print('Service Error: ', e) 

            # else: 
            #     print("! Waiting future !") 

    def inform_status(self):
        if self.lc.obs_lc != 0:
            print('Obstacle')
        
        elif self.lc.ovt_lc != 0:
            print('Overtaking')
        
        elif self.lc.global_path_lc != 0:
            print('Global path')
        
        elif self.forced_lc != 0:
            print('Forced lc')

    def publish_lc_final_direction(self):
        msg = Int16()
        msg.data = int(self.lc.lc_direction)
        if self.lc.lc_step >= 2:
            self.final_lc_direction_publisher.publish(msg)
        else:
            msg.data = int(0)
            self.final_lc_direction_publisher.publish(msg)

    def initialize_variable(self):
        if self.lc.path_id > 1000:
            self.forced_lc = self.ovt_lc = 0

    #Timer callback declaration
    def timer_callback(self):
        # switch = 3
        try:
            extended_lane = self.lc.extended_lane.count(self.lc.path_id)
            exception_lane = self.lc.exception_lane.count(self.lc.path_id)
            # print('##############################')

            if self.brk_pressure < 20:
                if not self.lc_status and self.lc.lc_direction_final == 0:
                    if self.lc.path_id > 1000:
                        self.lc.determine_avoiding()
                    
                    elif self.lc.path_id == 98:
                        if self.lc.remain_distance > 70 and self.control_status == 6:
                            self.lc.check_obstacle(front=True)

                        else:
                            self.lc.determine_avoiding()
                    
                    elif extended_lane == 1 or exception_lane == 1:
                        self.lc.determine_avoiding()

                    else:
                        if ((self.green_light == 0 or self.green_light == 2) and self.control_status == 4) or \
                            ((self.green_light == 0 or self.green_light == 2) and self.control_status == 6) or\
                                (self.green_light == 2 and self.go_triger == 0):
                                pass
                        
                        else:
                            self.lc.check_obstacle(front=True)
                            self.lc.check_ovt()
                            self.lc.determine_avoiding()
                
                else:
                    self.lc.determine_avoiding()

                self.lc.set_lc_direction(self.lc.global_path_lc, self.lc.obs_lc, self.lc.ovt_lc, self.forced_lc)

                if not self.lc_status and self.lc.lc_direction != 0:
                    self.lc.able_lc_direction()
                
                self.initialize_variable()
                self.lc.update(self.lc.lidar_front, self.lc.target_front, self.lc.target_rear, self.lc.version, self.lc.ego_vy, self.control_status)
                
                # if self.lc.path_id < 1000:
                #     cum_distance = self.lc.get_path_distance(self.lc.path_id)
                #     print('cum distance: ', cum_distance)
                #     print('remain distance: ', self.lc.remain_distance)
                
                # print('control status: ', self.control_status)
                # print('go triger: ', self.go_triger)
                # print('green light: ', self.green_light)
                # print('ego velocity: ', self.lc.ego_vy)
                # print('path id: ', self.lc.path_id)
                # print('remain distance: ', self.lc.remain_distance)
                # print('extended lane: ', extended_lane)
                # print('exception lane: ', exception_lane)
                # print('left able: ', self.lc.able_left)
                # print('right able: ', self.lc.able_right)
                # print('avoid count: ', self.lc.avoid_count)
                # print('lc ing count: ', self.lc.lc_ing_count)
                # print('obstacle: ', self.lc.obs_lc)
                # print('overtaking: ', self.lc.ovt_lc)
                # print('##############################')
                # print('waypoint_resest: ', self.lc.reset)

                self.request_lc_path()
                self.publish_lc_velocity(self.lc.lc_velocity)
                self.publish_lc_final_direction()
                self.publish_path_reset()
                self.publish_lc_distance(self.lc_distance)

        except Exception as e: 
            print('code error: ', e) 


def main(agrv=None):
    rclpy.init(args=None)

    dm = DecisionMaking()

    rclpy.spin(dm)
    dm.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()