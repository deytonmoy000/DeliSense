import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import collections
from datetime import datetime, timedelta
import random
import json

import concurrent.futures
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from queue import PriorityQueue

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import collections
from collections import deque
from datetime import datetime, timedelta
import random
import json

from collections import deque
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint

import concurrent.futures
import os
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
from queue import PriorityQueue
import time

from envs import get_visited_grid, get_shortest_path, get_shortest_path_length_rvs, generate_time_intervals, get_valid_grid, generate_rvs, OrderAssignmentEnv

'''
Global Parameters
'''
nDSV = 500
dsv_loc_val = -1000 # value to signify current loc of dsv
invalid_loc_val = -((2**31)-1)
max_valid_value = 720
size_dict = {1: [1, 1], 2: [1, 2], 4:[2, 2], 8: [2, 4], 16: [4, 4], 32:[4, 8]}

gridSize_info_dict = {1000:(54, 94, 5076, 54, 94, 5076), 500:(109, 188, 20492, 109, 188, 20492)}
ALPHA = 0.2
BETA = 2
GAMMA = 1
hourly_riders_dict = {0: 118, 1: 65, 2: 48, 3: 15, 4: 23, 5: 17, 6: 55, 7: 211, 8: 441, 9: 622, 10: 1547, 11: 2192, 12: 1764, 13: 1233, 14: 837, 15: 750, 16: 829, 17: 1104, 18: 1091, 19: 834, 20: 589, 21: 354, 22: 236, 23: 130}

'''
BASELINES
'''

class FastestDeliveryDispatch:  

    def __init__(self, orders, couriers, visited_grids):

        self.orders     = orders
        self.couriers    = couriers
        self.orders_f   = set()
        self.couriers_f  = set()
        self.od_f       = set()
        self.sparse = False
        self.sensing_cov  = visited_grids

        self.cur_min_LONG = 121.135
        self.cur_max_LONG = 121.8833
        self.cur_min_LAT = 31.0000
        self.cur_max_LAT = 31.3821
        self.block_size = 0.005  # block side length, representing 0.5 km

        self.delivery_quotient_mat = self.gen_quotient()

    def reset(self):
        self.orders_f     = set()

        self.delivery_quotient_mat = self.update_quotient()

    def gen_quotient(self):
        delivery_quotient_mat = [[] for i in range(len(self.orders))]
        for i, order in enumerate(self.orders):
            for j, courier in enumerate(self.couriers):
                if (i, j) in self.od_f:
                    delivery_quotient_mat[i].append(-1e6)
                    continue

                order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
                order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
                order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
                order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
                courier_long_id, courier_lat_id = courier['location']

                S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
                S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
                delivery_time = S_p + S_d
                time_saved = order['deadline'] - (S_p + S_d)
                courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)

                delivery_quotient = -(len(courier_path_to_dest) + 1e-8)

                delivery_quotient_mat[i].append(delivery_quotient)

        return np.array(delivery_quotient_mat)

    def get_quotient_mat(self):
        dispatch_quotient = self.delivery_quotient_mat
        quotient_list = []
        for i in range(dispatch_quotient.shape[0]):
            for j in range(dispatch_quotient.shape[1]):
                quotient_list.append([dispatch_quotient[i, j], i, j])
        return quotient_list

    def update_quotient(self):
        delivery_quotient_mat = [[] for i in range(len(self.orders))]
        for i, order in enumerate(self.orders):
            if i in self.orders_f:
                delivery_quotient_mat[i].append(1e6)
                continue
            for j, courier in enumerate(self.couriers):
                if j in self.couriers_f:
                    delivery_quotient_mat[i].append(1e6)
                    continue

                if (i, j) in self.od_f:
                    delivery_quotient_mat[i].append(1e6)
                    continue

                order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
                order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
                order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
                order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
                courier_long_id, courier_lat_id = courier['location']

                S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
                S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
                delivery_time = S_p + S_d
                time_saved = order['deadline'] - (S_p + S_d)
                courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)

                delivery_quotient = -(len(courier_path_to_dest) + 1e-8)

                delivery_quotient_mat[i].append(delivery_quotient)

        return np.array(delivery_quotient_mat)

    def get_best_dispatch(self):
        dispatch_quotient = self.delivery_quotient_mat

        action, reward = np.argmax(dispatch_quotient), np.max(dispatch_quotient)
        order_id, courier_id = divmod(action, len(self.couriers))

        return self.orders[order_id]['id'], self.couriers[courier_id]['id'], reward

    def dispatch(self):
        dispatch_quotient = self.delivery_quotient_mat

        action, reward = np.argmax(dispatch_quotient), np.max(dispatch_quotient)
        if reward == -1e6:
            return (-1, -1, -1, -1)
        order_id, courier_id = divmod(action, len(self.couriers))

        return self.assign(order_id, courier_id)

    def assign(self, order_id, courier_id):

        self.delivery_quotient_mat[:,courier_id] = -1e6
        self.delivery_quotient_mat[order_id,:] = -1e6
        self.couriers_f.add(courier_id)
        self.orders_f.add(order_id)
        self.od_f.add((order_id, courier_id))

        order = self.orders[order_id]
        courier = self.couriers[courier_id]
        order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
        order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
        order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
        order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
        courier_long_id, courier_lat_id = courier['location']

        S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
        S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))

        courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)
        courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
        new_locs = len(courier_path_to_dest_set - self.sensing_cov)

        self.sensing_cov = self.sensing_cov.union(courier_path_to_dest_set)
        self.update_quotient()
        return (order_id, courier_id, order['id'], courier['id'])

        dispatch_quotient = self.ranking_quotient_mat*self.cov_quotient_mat

class BestSensingDispatch:  

    def __init__(self, orders, couriers, visited_grids):

        self.orders     = orders
        self.couriers    = couriers
        self.orders_f   = set()
        self.couriers_f  = set()
        self.od_f       = set()
        self.sparse = False
        self.sensing_cov  = visited_grids

        self.cur_min_LONG = 121.135
        self.cur_max_LONG = 121.8833
        self.cur_min_LAT = 31.0000
        self.cur_max_LAT = 31.3821
        self.block_size = 0.005  # block side length, representing 0.5 km

        self.sensing_quotient_mat = self.gen_quotient()

    def reset(self):
        self.orders_f     = set()

        self.sensing_quotient_mat = self.update_quotient()

    def gen_quotient(self):
        sensing_quotient_mat = [[] for i in range(len(self.orders))]
        for i, order in enumerate(self.orders):
            for j, courier in enumerate(self.couriers):
                if (i, j) in self.od_f:
                    sensing_quotient_mat[i].append(-1e6)
                    continue

                order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
                order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
                order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
                order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
                courier_long_id, courier_lat_id = courier['location']

                S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
                S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
                delivery_time = S_p + S_d
                time_saved = order['deadline'] - (S_p + S_d)
                courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)

                courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
                n_new_grids = len(courier_path_to_dest_set - self.sensing_cov)
                sensing_reward = n_new_grids+1
                sensing_quotient_mat[i].append(sensing_reward)

        return np.array(sensing_quotient_mat)

    def get_quotient_mat(self):
        dispatch_quotient = self.sensing_quotient_mat
        quotient_list = []
        for i in range(dispatch_quotient.shape[0]):
            for j in range(dispatch_quotient.shape[1]):
                quotient_list.append([dispatch_quotient[i, j], i, j])
        return quotient_list

    def update_quotient(self):
        sensing_quotient_mat = [[] for i in range(len(self.orders))]
        for i, order in enumerate(self.orders):
            if i in self.orders_f:
                sensing_quotient_mat[i].append(-1e6)
                continue
            for j, courier in enumerate(self.couriers):
                if j in self.couriers_f:
                    sensing_quotient_mat[i].append(-1e6)
                    continue

                if (i, j) in self.od_f:
                    sensing_quotient_mat[i].append(-1e6)
                    continue

                order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
                order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
                order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
                order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
                courier_long_id, courier_lat_id = courier['location']

                S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
                S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
                delivery_time = S_p + S_d
                time_saved = order['deadline'] - (S_p + S_d)
                courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)

                courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
                n_new_grids = len(courier_path_to_dest_set - self.sensing_cov)

                sensing_reward = n_new_grids+1

                sensing_quotient_mat[i].append(sensing_reward)

        return np.array(sensing_quotient_mat)

    def get_best_dispatch(self):
        dispatch_quotient = self.sensing_quotient_mat

        action, reward = np.argmax(dispatch_quotient), np.max(dispatch_quotient)
        order_id, courier_id = divmod(action, len(self.couriers))

        return self.orders[order_id]['id'], self.couriers[courier_id]['id'], reward

    def dispatch(self):
        dispatch_quotient = self.sensing_quotient_mat

        action, reward = np.argmax(dispatch_quotient), np.max(dispatch_quotient)
        if reward == -1e6:
            return (-1, -1, -1, -1)
        order_id, courier_id = divmod(action, len(self.couriers))

        return self.assign(order_id, courier_id)

    def assign(self, order_id, courier_id):

        self.sensing_quotient_mat[:,courier_id] = -1e6
        self.sensing_quotient_mat[order_id,:] = -1e6
        self.couriers_f.add(courier_id)
        self.orders_f.add(order_id)
        self.od_f.add((order_id, courier_id))

        order = self.orders[order_id]
        courier = self.couriers[courier_id]
        order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
        order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
        order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
        order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
        courier_long_id, courier_lat_id = courier['location']

        S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
        S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))

        courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)
        courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
        new_locs = len(courier_path_to_dest_set - self.sensing_cov)

        self.sensing_cov = self.sensing_cov.union(courier_path_to_dest_set)
        self.update_quotient()

        return (order_id, courier_id, order['id'], courier['id'])
        dispatch_quotient = self.ranking_quotient_mat*self.cov_quotient_mat

class DeliverySensingEffiencyDispatch:  

    def __init__(self, orders, couriers, visited_grids):

        self.orders     = orders
        self.couriers    = couriers
        self.orders_f   = set()
        self.couriers_f  = set()
        self.od_f       = set()
        self.sparse = False
        self.sensing_cov  = visited_grids

        self.cur_min_LONG = 121.135
        self.cur_max_LONG = 121.8833
        self.cur_min_LAT = 31.0000
        self.cur_max_LAT = 31.3821
        self.block_size = 0.005  # block side length, representing 0.5 km

        self.delivery_quotient_mat, self.sensing_quotient_mat = self.gen_quotient()

    def reset(self):
        self.orders_f     = set()

        self.delivery_quotient_mat, self.sensing_quotient_mat = self.update_quotient()

    def gen_quotient(self):
        delivery_quotient_mat = [[] for i in range(len(self.orders))]
        sensing_quotient_mat = [[] for i in range(len(self.orders))]
        for i, order in enumerate(self.orders):
            for j, courier in enumerate(self.couriers):
                if (i, j) in self.od_f:
                    delivery_quotient_mat[i].append(1)
                    sensing_quotient_mat[i].append(-1e6)
                    continue

                order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
                order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
                order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
                order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
                courier_long_id, courier_lat_id = courier['location']

                S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
                S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
                delivery_time = S_p + S_d
                time_saved = order['deadline'] - (S_p + S_d)
                courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)

                delivery_quotient = len(courier_path_to_dest)+1
                delivery_quotient_mat[i].append(delivery_quotient)

                courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
                n_new_grids = len(courier_path_to_dest_set - self.sensing_cov)

                sensing_reward = n_new_grids+1
                sensing_quotient_mat[i].append(sensing_reward)

        return np.array(delivery_quotient_mat), np.array(sensing_quotient_mat)

    def get_quotient_mat(self):
        dispatch_quotient = self.sensing_quotient_mat/self.delivery_quotient_mat
        quotient_list = []
        for i in range(dispatch_quotient.shape[0]):
            for j in range(dispatch_quotient.shape[1]):
                quotient_list.append([dispatch_quotient[i, j], i, j])
        return quotient_list

    def update_quotient(self):
        delivery_quotient_mat = [[] for i in range(len(self.orders))]
        sensing_quotient_mat = [[] for i in range(len(self.orders))]
        for i, order in enumerate(self.orders):
            if i in self.orders_f:
                delivery_quotient_mat[i].append(1)
                sensing_quotient_mat[i].append(-1e6)
                continue
            for j, courier in enumerate(self.couriers):
                if j in self.couriers_f:
                    delivery_quotient_mat[i].append(1)
                    sensing_quotient_mat[i].append(-1e6)
                    continue

                if (i, j) in self.od_f:
                    delivery_quotient_mat[i].append(1)
                    sensing_quotient_mat[i].append(-1e6)
                    continue

                order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
                order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
                order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
                order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
                courier_long_id, courier_lat_id = courier['location']

                S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
                S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
                delivery_time = S_p + S_d
                time_saved = order['deadline'] - (S_p + S_d)
                courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)

                delivery_quotient = len(courier_path_to_dest)+1
                delivery_quotient_mat[i].append(delivery_quotient)

                courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
                n_new_grids = len(courier_path_to_dest_set - self.sensing_cov)

                sensing_reward = n_new_grids+1
                sensing_quotient_mat[i].append(sensing_reward)

        return np.array(delivery_quotient_mat), np.array(sensing_quotient_mat)

    def get_best_dispatch(self):
        dispatch_quotient = self.sensing_quotient_mat/self.delivery_quotient_mat

        action, reward = np.argmax(dispatch_quotient), np.max(dispatch_quotient)
        order_id, courier_id = divmod(action, len(self.couriers))

        return self.orders[order_id]['id'], self.couriers[courier_id]['id'], reward

    def dispatch(self):
        dispatch_quotient = self.delivery_quotient_mat*self.sensing_quotient_mat

        action, reward = np.argmax(dispatch_quotient), np.max(dispatch_quotient)
        if reward == -1e6:
            return (-1, -1, -1, -1)
        order_id, courier_id = divmod(action, len(self.couriers))

        return self.assign(order_id, courier_id)

    def assign(self, order_id, courier_id):

        self.delivery_quotient_mat[:,courier_id] = 1
        self.sensing_quotient_mat[:,courier_id] = -1e6
        self.delivery_quotient_mat[order_id,:] = 1
        self.sensing_quotient_mat[order_id,:] = -1e6
        self.couriers_f.add(courier_id)
        self.orders_f.add(order_id)
        self.od_f.add((order_id, courier_id))

        order = self.orders[order_id]
        courier = self.couriers[courier_id]
        order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
        order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
        order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
        order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
        courier_long_id, courier_lat_id = courier['location']

        S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
        S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))

        courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)
        courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
        new_locs = len(courier_path_to_dest_set - self.sensing_cov)

        self.sensing_cov = self.sensing_cov.union(courier_path_to_dest_set)
        self.update_quotient()

        return (order_id, courier_id, order['id'], courier['id'])
        dispatch_quotient = self.ranking_quotient_mat*self.cov_quotient_mat

class LSTAloc:  

    def __init__(self, orders, couriers, visited_grids):

        self.orders     = orders
        self.couriers    = couriers
        self.orders_f   = set()
        self.couriers_f  = set()
        self.od_f       = set()
        self.sparse = False
        self.sensing_cov  = visited_grids

        self.cur_min_LONG = 121.135
        self.cur_max_LONG = 121.8833
        self.cur_min_LAT = 31.0000
        self.cur_max_LAT = 31.3821
        self.block_size = 0.005  # block side length, representing 0.5 km

        self.delivery_incentive_quotient_mat, self.sensing_quotient_mat = self.gen_quotient()

    def reset(self):
        self.orders_f     = set()

        self.delivery_incentive_quotient_mat, self.sensing_quotient_mat = self.update_quotient()

    def gen_quotient(self):
        delivery_incentive_quotient_mat = [[] for i in range(len(self.orders))]
        sensing_quotient_mat = [[] for i in range(len(self.orders))]
        for i, order in enumerate(self.orders):
            for j, courier in enumerate(self.couriers):
                if (i, j) in self.od_f:
                    delivery_incentive_quotient_mat[i].append(1)
                    sensing_quotient_mat[i].append(-1e6)
                    continue

                order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
                order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
                order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
                order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
                courier_long_id, courier_lat_id = courier['location']

                S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
                S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
                delivery_time = S_p + S_d
                
                if courier['incentive']:
                    delivery_incentive = np.sum(courier['incentive'])
                else:
                    delivery_incentive = 0.1
                
                delivery_quotient = 1/delivery_incentive

                delivery_incentive_quotient_mat[i].append(delivery_quotient)

                sensing_reward = 1/(delivery_time + 1e-8)
                sensing_quotient_mat[i].append(sensing_reward)

        return np.array(delivery_incentive_quotient_mat), np.array(sensing_quotient_mat)

    def get_quotient_mat(self):
        dispatch_quotient = self.sensing_quotient_mat/self.delivery_incentive_quotient_mat
        quotient_list = []
        for i in range(dispatch_quotient.shape[0]):
            for j in range(dispatch_quotient.shape[1]):
                quotient_list.append([dispatch_quotient[i, j], i, j])
        return quotient_list

    def update_quotient(self):
        delivery_incentive_quotient_mat = [[] for i in range(len(self.orders))]
        sensing_quotient_mat = [[] for i in range(len(self.orders))]
        for i, order in enumerate(self.orders):
            if i in self.orders_f:
                delivery_incentive_quotient_mat[i].append(1)
                sensing_quotient_mat[i].append(-1e6)
                continue
            for j, courier in enumerate(self.couriers):
                if j in self.couriers_f:
                    delivery_incentive_quotient_mat[i].append(1)
                    sensing_quotient_mat[i].append(-1e6)
                    continue

                if (i, j) in self.od_f:
                    delivery_incentive_quotient_mat[i].append(1)
                    sensing_quotient_mat[i].append(-1e6)
                    continue

                order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
                order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
                order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
                order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
                courier_long_id, courier_lat_id = courier['location']

                S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
                S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
                delivery_time = S_p + S_d
                
                if courier['incentive']:
                    delivery_incentive = np.sum(courier['incentive'])
                else:
                    delivery_incentive = 0.001
                
                delivery_quotient = 1/delivery_incentive

                delivery_incentive_quotient_mat[i].append(delivery_quotient)

                sensing_reward = 1/(delivery_time + 1e-8)
                sensing_quotient_mat[i].append(sensing_reward)

        return np.array(delivery_incentive_quotient_mat), np.array(sensing_quotient_mat)

    def get_best_dispatch(self):
        dispatch_quotient = self.sensing_quotient_mat/self.delivery_incentive_quotient_mat

        action, reward = np.argmax(dispatch_quotient), np.max(dispatch_quotient)
        order_id, courier_id = divmod(action, len(self.couriers))

        return self.orders[order_id]['id'], self.couriers[courier_id]['id'], reward

    def dispatch(self):
        dispatch_quotient = self.delivery_incentive_quotient_mat*self.sensing_quotient_mat

        action, reward = np.argmax(dispatch_quotient), np.max(dispatch_quotient)
        if reward == -1e6:
            return (-1, -1, -1, -1)
        order_id, courier_id = divmod(action, len(self.couriers))

        return self.assign(order_id, courier_id)

    def assign(self, order_id, courier_id):

        self.delivery_incentive_quotient_mat[:,courier_id] = 1
        self.sensing_quotient_mat[:,courier_id] = -1e6
        self.delivery_incentive_quotient_mat[order_id,:] = 1
        self.sensing_quotient_mat[order_id,:] = -1e6
        self.couriers_f.add(courier_id)
        self.orders_f.add(order_id)
        self.od_f.add((order_id, courier_id))

        order = self.orders[order_id]
        courier = self.couriers[courier_id]
        order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
        order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
        order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
        order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
        courier_long_id, courier_lat_id = courier['location']

        S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
        S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))

        courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)
        courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
        new_locs = len(courier_path_to_dest_set - self.sensing_cov)

        self.sensing_cov = self.sensing_cov.union(courier_path_to_dest_set)
        self.update_quotient()

        return (order_id, courier_id, order['id'], courier['id'])
        dispatch_quotient = self.ranking_quotient_mat*self.cov_quotient_mat

class AJRP:  

    def __init__(self, orders, couriers, visited_grids):

        self.orders     = orders
        self.couriers    = couriers
        self.orders_f   = set()
        self.couriers_f  = set()
        self.od_f       = set()
        self.sparse = False
        self.sensing_cov  = visited_grids

        self.cur_min_LONG = 121.135
        self.cur_max_LONG = 121.8833
        self.cur_min_LAT = 31.0000
        self.cur_max_LAT = 31.3821
        self.block_size = 0.005  # block side length, representing 0.5 km

        self.delivery_quotient_mat, self.sensing_quotient_mat = self.gen_quotient()

    def reset(self):
        self.orders_f     = set()

        self.delivery_quotient_mat, self.sensing_quotient_mat = self.update_quotient()

    def gen_quotient(self):
        delivery_quotient_mat = [[] for i in range(len(self.orders))]
        sensing_quotient_mat = [[] for i in range(len(self.orders))]
        for i, order in enumerate(self.orders):
            for j, courier in enumerate(self.couriers):
                if (i, j) in self.od_f:
                    delivery_quotient_mat[i].append(1)
                    sensing_quotient_mat[i].append(-1e6)
                    continue

                order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
                order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
                order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
                order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
                courier_long_id, courier_lat_id = courier['location']

                S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
                S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
                delivery_time = S_p + S_d
                time_saved = order['deadline'] - (S_p + S_d)
                courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)

                delivery_quotient = len(courier_path_to_dest)+1
                delivery_quotient_mat[i].append(delivery_quotient)

                courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
                n_new_grids = len(courier_path_to_dest_set - self.sensing_cov)

                sensing_reward = n_new_grids+1
                sensing_quotient_mat[i].append(sensing_reward)

        return np.array(delivery_quotient_mat), np.array(sensing_quotient_mat)

    def get_quotient_mat(self):
        dispatch_quotient = self.sensing_quotient_mat/self.delivery_quotient_mat
        quotient_list = []
        for i in range(dispatch_quotient.shape[0]):
            for j in range(dispatch_quotient.shape[1]):
                quotient_list.append([dispatch_quotient[i, j], i, j])
        return quotient_list

    def update_quotient(self):
        delivery_quotient_mat = [[] for i in range(len(self.orders))]
        sensing_quotient_mat = [[] for i in range(len(self.orders))]
        for i, order in enumerate(self.orders):
            if i in self.orders_f:
                delivery_quotient_mat[i].append(1)
                sensing_quotient_mat[i].append(-1e6)
                continue
            for j, courier in enumerate(self.couriers):
                if j in self.couriers_f:
                    delivery_quotient_mat[i].append(1)
                    sensing_quotient_mat[i].append(-1e6)
                    continue

                if (i, j) in self.od_f:
                    delivery_quotient_mat[i].append(1)
                    sensing_quotient_mat[i].append(-1e6)
                    continue

                order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
                order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
                order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
                order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
                courier_long_id, courier_lat_id = courier['location']

                S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
                S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
                delivery_time = S_p + S_d
                time_saved = order['deadline'] - (S_p + S_d)
                courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)

                delivery_quotient = len(courier_path_to_dest)+1
                delivery_quotient_mat[i].append(delivery_quotient)

                courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
                n_new_grids = len(courier_path_to_dest_set - self.sensing_cov)

                sensing_reward = n_new_grids+1
                sensing_quotient_mat[i].append(sensing_reward)

        return np.array(delivery_quotient_mat), np.array(sensing_quotient_mat)

    def get_best_dispatch(self):
        dispatch_quotient = self.sensing_quotient_mat/self.delivery_quotient_mat

        action, reward = np.argmax(dispatch_quotient), np.max(dispatch_quotient)
        order_id, courier_id = divmod(action, len(self.couriers))

        return self.orders[order_id]['id'], self.couriers[courier_id]['id'], reward

    def dispatch(self):
        dispatch_quotient = self.delivery_quotient_mat*self.sensing_quotient_mat

        action, reward = np.argmax(dispatch_quotient), np.max(dispatch_quotient)
        if reward == -1e6:
            return (-1, -1, -1, -1)
        order_id, courier_id = divmod(action, len(self.couriers))

        return self.assign(order_id, courier_id)

    def assign(self, order_id, courier_id):

        self.delivery_quotient_mat[:,courier_id] = 1
        self.sensing_quotient_mat[:,courier_id] = -1e6
        self.delivery_quotient_mat[order_id,:] = 1
        self.sensing_quotient_mat[order_id,:] = -1e6
        self.couriers_f.add(courier_id)
        self.orders_f.add(order_id)
        self.od_f.add((order_id, courier_id))

        order = self.orders[order_id]
        courier = self.couriers[courier_id]
        order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
        order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
        order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
        order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
        courier_long_id, courier_lat_id = courier['location']

        S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
        S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))

        courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)
        courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
        new_locs = len(courier_path_to_dest_set - self.sensing_cov)

        self.sensing_cov = self.sensing_cov.union(courier_path_to_dest_set)
        self.update_quotient()

        return (order_id, courier_id, order['id'], courier['id'])
        dispatch_quotient = self.ranking_quotient_mat*self.cov_quotient_mat


def OrderDispatch(orders, couriers, visited_grids, name):
    

    if name[:3] == "fdd":
        objective = FastestDeliveryDispatch(orders.copy(), couriers.copy(), visited_grids.copy())
    elif name[:3] == "bsd":
        objective = BestSensingDispatch(orders.copy(), couriers.copy(), visited_grids.copy())
    elif name[:3] == "dsd":
        objective = DeliverySensingEffiencyDispatch(orders.copy(), couriers.copy(), visited_grids.copy())
    elif name[:4] == "lsta":
        objective = LSTAloc(orders.copy(), couriers.copy(), visited_grids.copy())
    elif name[:4] == "ajrp":
        objective = AJRP(orders.copy(), couriers.copy(), visited_grids.copy())

    D = []
    if len(orders) == 0:
        return D
    couriers_assigned = set()

    if len(couriers) == 0:
        return []

    while len(objective.orders_f) != len(orders):

        o, d, o_id, d_id = objective.dispatch()
        if o == -1 and d == -1:
            break
        D.append((o, d, o_id, d_id))
        couriers_assigned.add(d)

        if len(couriers_assigned) == len(couriers):
            break
    objective.reset()   
    return D

if __name__ == "__main__":

    name = sys.argv[1]
    n_rvs = int(sys.argv[2]) # Number of RVs
    print(f"Initializing run for {name} with {n_rvs} RVs")

    rvs = generate_rvs(n_rvs)
    visited_grids = get_visited_grid(500)
    interval = 60

    df = pd.read_csv('data/order_data_test_2020_1013-1014_zhonghuan.csv')
    df['create'] = pd.to_datetime(df['create'])

    results = []

    env = OrderAssignmentEnv(df, 0, rvs, visited_grids, name, interval=interval, n_couriers=hourly_riders_dict[0], n_rvs=n_rvs)
    overdue_cnt = 0

    for h in range(24):
        print("Initiating New OrderAssignmentEnv")
        env = OrderAssignmentEnv(df, 0, rvs, visited_grids, name, interval=interval, n_couriers=hourly_riders_dict[h], n_rvs=n_rvs)

        env.t = 0
        env.time_intervals = generate_time_intervals(h, interval)

        couriers = []
        start_time = time.time()
        for i in range(int(60*(60/env.interval))):
            print(f"\n<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>\n")
            # Human Driver Assignments
            env.reset()
            orders_rem = [order for order in env.orders if order['delivery_time'] <= 0]
            orders_rem_old = orders_rem
            print(f"Orders Remaining At Start: {len(orders_rem)}")

            couriers = [courier for courier in env.couriers if courier['available_in'] <= 0]

            print(f"Available Drivers Before Decrement: {len(couriers)}")
            n_courier = env.decrement_courier_availability()

            couriers_rem = [courier for courier in env.couriers if courier['available_in'] <= 0]

            couriers_rv = [courier for courier in env.rvs if courier['available_in'] <= 0]
            print(f"Available RVs Before Decrement: {len(couriers)}")
            env.decrement_rv_availability()
            couriers_rem_rv = [courier for courier in env.rvs if courier['available_in'] <= 0]
            print(f"Available RVs After Decrement: {len(couriers)}")

            while len(orders_rem) != 0:
                if name[:4] == "lsta" or name[:4] == "ajrp":
                    if len(couriers_rem):
                        all_couriers = couriers_rem
                    else:
                        all_couriers = couriers_rem_rv
                else:
                    all_couriers = couriers_rem + couriers_rem_rv   
                orders_rem = [order for order in env.orders if order['delivery_time'] <= 0]

                orders_rem_old = orders_rem
                if len(orders_rem) == 0:
                    print("No Order Remaining For Human...")
                    break

                if len(all_couriers) == 0:
                    print("No Driver Available")
                    break

                D = OrderDispatch(orders_rem, all_couriers, visited_grids, name)

                total_rew = 0
                orders_rem = [order for order in env.orders if order['delivery_time'] <= 0]

                for o, d, o_id, d_id in D:

                    if len(couriers_rem):
                        d_actual = d//len(couriers_rem)
                    else:
                        d_actual=1
                    if d_actual == 0:
                        reward, done = env.update(o_id, d_id)
                    else:
                        reward, done = env.update_rvs(o_id, d_id)

                    orders_rem = [order for order in env.orders if order['delivery_time'] <= 0]
                    couriers_TMP = [courier for courier in env.couriers if courier['available_in'] <= 0]
                    couriers_rvs_TMP = [courier for courier in env.rvs if courier['available_in'] <= 0]

                    if reward < 0:
                        overdue_cnt += 1

                orders_rem = [order for order in env.orders if order['delivery_time'] <= 0]
                couriers_rem = [courier for courier in env.couriers if courier['available_in'] <= 0]

                print(f"\nCompleted Assignments >> Time({env.time_intervals[env.t][0]} - {env.time_intervals[env.t][1]}) (Assigned {len(D)} Orders) (Remaining {len(orders_rem)} Orders) (Reamining Drivers: {len(couriers_rem)})\n")
                visited_grids = env.visited_grids

                orders_rem = [order for order in env.orders if order['delivery_time'] <= 0]

                couriers_rem = [courier for courier in env.couriers if courier['available_in'] <= 0]

                couriers_rem_rv = [courier for courier in env.rvs if courier['available_in'] <= 0]

            #################################################################
            print(f"\n ASSIGNMENTS COMPLETE\n")

            orders_overdue = [order for order in env.orders if order['delivery_time'] > order['deadline']]
            overdue_cnt += len(orders_overdue)

            for vid in range(len(env.rvs)):
                if name[-3:] == "tsm":
                    env.rv_sense_tsmtc(vid)
                elif name[-3:] == "rsn":
                    env.rv_sense_reassign(vid)
                elif name[-3:] == "sdp":
                    env.rv_sense_sdpr(vid)
                else:
                    env.rv_sense_sdpr(vid)

            env.t += 1    

        end_time = time.time()
        execution_time = end_time - start_time

        # Hourly Metrics
        new_locs = len(visited_grids) - env.total_cov
        orders_rem = len([order for order in env.orders if order['delivery_time'] <= 0])
        overdue_cnt += orders_rem
        dii_all = []
        di_all = []
        hourly_incentive = 0
        for courier in env.couriers:
            if courier['incentive_index']:
                dii_all.append(np.mean(courier['incentive_index']))
            if courier['incentive']:
                di_all.append(np.sum(courier['incentive']))
                hourly_incentive += np.sum(courier['incentive'])
        avg_di = hourly_incentive/hourly_riders_dict[h]
        avg_dii = np.mean(dii_all)

    # Append the results to the list

        print(f"\n-x-x-x-x-x-HOURLY RESULT-x-x-x-x-x-\n")
        print(f"Hour ({h}) >> Total Sensing Coverage: {len(visited_grids)}")
        print(f"Hour ({h}) >> New Locations Sensing Coverage: {new_locs}")
        print(f"Hour ({h}) >> Order Overdue: {overdue_cnt}")
        print(f"Hour ({h}) >> Average Driving Incentive : {avg_di}")   
        print(f"Hour ({h}) >> Average Driving Incentive Index: {avg_dii}")   
        print(f"\n-x-x-x-x-x-x-x-x-x-x-\n")

        results.append({
            "Model": name,
            "Hour": h,
            "nRVs": n_rvs,
            "Total_Sensing_Coverage": len(visited_grids),
            "New_Locations_Sensing_Coverage": new_locs,
            "Order_Overdue": overdue_cnt,
            "Average_Driving_Incentive": avg_di, 
            "Average_Driving_Incentive_Index": avg_dii,
            "executionTime": execution_time
        })
        res_df = pd.DataFrame(results)
        fname=f"results/MR-KSubMod-Result_{name}_{n_rvs}.csv"
        res_df.to_csv(fname)

        overdue_cnt = 0
        visited_grids = get_visited_grid(500)

    res_df = pd.DataFrame(results)
    fname=f"results/MR-KSubMod-Result_{name}_{n_rvs}.csv"
    res_df.to_csv(fname)
    print("Run Complete")

