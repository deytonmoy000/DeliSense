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

from collections import deque
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, Flatten, Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import ModelCheckpoint

os.environ['CUDA_VISIBLE_DEVICES'] = ''

from queue import PriorityQueue
from collections import deque
from tensorflow.keras import backend as K
import gc

hourly_riders_dict = {0: 118, 1: 65, 2: 48, 3: 15, 4: 23, 5: 17, 6: 55, 7: 211, 8: 441, 9: 622, 10: 1547, 11: 2192, 12: 1764, 13: 1233, 14: 837, 15: 750, 16: 829, 17: 1104, 18: 1091, 19: 834, 20: 589, 21: 354, 22: 236, 23: 130}
gridSize_info_dict = {1000:(38, 74, 2812, 38, 74, 2812), 500:(76, 148, 11248, 76, 148, 11248)}
dsv_loc_val = -1000 # value to signify current loc of dsv
invalid_loc_val = -((2**31)-1)
max_valid_value = 720

def pos_to_coord(grid_size, pos):
    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 

    return (pos//n_cols, pos%n_cols)

def coord_to_pos(grid_size, lat, lon):
    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 

    return n_cols*lat + lon

def is_above_line_pt(lon, lat):

    x1, y1 = 147, 52
    x2, y2 = 86, 0
    m = (y2 - y1) / (x2 - x1)

    return lat > m * (lon - x1) + y1

def get_valid_grid(grid_size):
    valid_cells = set()
    min_LONG_ID, max_LONG_ID, min_LAT_ID, max_LAT_ID = 0, gridSize_info_dict[grid_size][1], 0, gridSize_info_dict[grid_size][0]
    for lon in range(min_LONG_ID, max_LONG_ID+1):
        for lat in range(min_LAT_ID, max_LAT_ID+1):
            if not is_above_line_pt(lon, lat) or lon >= max_LONG_ID or lat >= max_LAT_ID:
                continue
            valid_cells.add((lon, lat))
    return valid_cells

def get_visited_grid(grid_size):
    visited_cells = set()
    fname_valid_grids = f'data/map/visited_{str(int(grid_size))}m_shanghai.txt'
    max_LONG_ID, max_LAT_ID = gridSize_info_dict[grid_size][1], gridSize_info_dict[grid_size][0]

    with open(fname_valid_grids, "r") as file:
        for line in file:
            x, y = map(int, line.split(','))
            visited_cells.add((x, y))
    file.close()

    visited_cells_cleaned = visited_cells.copy()
    for cell in visited_cells:
        lon, lat = cell
        if not is_above_line_pt(lon, lat) or lon >= max_LONG_ID or lat >= max_LAT_ID:

            visited_cells_cleaned.remove((lon, lat))

    return visited_cells_cleaned

def generate_time_intervals(h, interval, tf=1):

    start_date = datetime(2024, 5, 16, h, 0, 0)
    end_date = datetime(2024, 5, 16, h, 59, 59)  # Change this to the end date you need

    window_end_int = timedelta(seconds=interval-1)
    current_date = start_date
    time_intervals = []
    while current_date <= end_date:
        window_end_date = current_date + window_end_int
        s, e = pd.to_datetime(current_date.strftime('%Y-%m-%d %H:%M:%S')),pd.to_datetime(window_end_date.strftime('%Y-%m-%d %H:%M:%S'))
        time_intervals.append([s, e])
        next_date_int = timedelta(seconds=interval)
        current_date = current_date + next_date_int

    return time_intervals

def get_shortest_path(lonStart, latStart, lonPickup, latPickup, lonDest, latDest):
    path = [[lonStart, latStart]]
    lonS, latS, lonP, latP, lonD, latD = lonStart, latStart, lonPickup, latPickup, lonDest, latDest

    dirLong = int((lonPickup - lonStart)/(abs(lonPickup - lonStart))) if abs(lonPickup - lonStart) > 0  else 0
    dirLat = int((latPickup - latStart)/(abs(latPickup - latStart))) if abs(latPickup - latStart) > 0  else 0

    while (abs(latP - latS)!=0 and abs(lonS - lonP)!=0):
        latS += dirLat
        lonS += dirLong

        path.append([lonS, latS])

    while abs(latP - latS)!=0:
        latS += dirLat
        path.append([lonS, latS])
    while abs(lonP - lonS)!=0:
        lonS += dirLong
        path.append([lonS, latS])

    # Pickup to Destination
    dirLong = int((lonDest - lonPickup)/(abs(lonDest - lonPickup))) if abs(lonDest - lonPickup) > 0  else 0
    dirLat = int((latDest - latPickup)/(abs(latDest - latPickup))) if abs(latDest - latPickup) > 0  else 0

    while (abs(latD - latP)!=0 and abs(lonP - lonD)!=0):
        latP += dirLat
        lonP += dirLong

        path.append([lonP, latP])

    while abs(latD - latP)!=0:
        latP += dirLat
        path.append([lonP, latP])
    while abs(lonD - lonP)!=0:
        lonP += dirLong
        path.append([lonP, latP])

    return path

def get_shortest_path_length_rvs(lonStart, latStart, lonPickup, latPickup, lonDest, latDest):
    lonS, latS, lonP, latP, lonD, latD = lonStart, latStart, lonPickup, latPickup, lonDest, latDest

    if lonP == -1 and latP == -1 and lonD == -1 and latD == -1:
        return 0
    elif lonP == -1 and latP == -1:
        return max(abs(latD - latS) , abs(lonD - lonS))
    else:
        S_p = max(abs(latP - latS) , abs(lonP - lonS))
        S_d = max(abs(latD - latS) , abs(lonD - lonS))
        return S_p+S_d

def generate_rvs(n_rv, grid_size=500):

    num_couriers = n_rv
    valid_grids = get_valid_grid(grid_size)
    couriers = []
    random.seed(35)
    for i in range(num_couriers):
        courier = {
            'id': i,
            'location': random.choice(list(valid_grids)),
            'pickup_loc': [-1, -1],
            'dest_loc': [-1, -1],
            'incentive_index': [],
            'incentive': [],
            'path_to_dest': deque([]),
            'deadline': None,
            'available_in': 0  # when rider becomes available, could be calculated or updated over time
        }
        couriers.append(courier)

    return couriers

def generate_neighbors_list_wBorders(n, grid_size, alg, tf, km_time):

    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size] 
    print(f"generate_neighbors_list_wBorders --> n_rows:{n_rows}, n_cols:{n_cols};  n_rows_valid:{n_rows_valid}, n_cols_valid:{n_cols_valid}")

    ts = tf//km_time
    valid_cells = get_valid_grid(grid_size)
    valid_locs = set()
    for lon, lat in valid_cells:
        pos = n_cols*lat + lon
        valid_locs.add(pos)
    div_coords = {}

    neighbors = collections.defaultdict(list)
    directions = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
    area_nrows, area_ncols = 1, 1
    area_width, area_height = n_cols_valid//area_ncols, n_rows_valid//area_nrows
    mat = get_default_mat2D_DQN(grid_size)
    valid_div_pos = []
    pos_to_div = {}

    for i in range(area_nrows):
        for j in range(area_ncols):
            div_pos = set()
            div_id = i*area_ncols + j
            start_row, end_row = i*area_height, (i+1)*area_height
            start_col, end_col = j*area_width, (j+1)*area_width

            if j == area_ncols-1: 
                end_col = n_cols_valid
            if i == area_nrows-1:
                end_row = n_rows_valid
            for lat in range(start_row, end_row):
                for lon in range(start_col, end_col):
                    if (lat, lon) in valid_cells:
                        pos = n_cols*lat + lon
                        pos_to_div[(lat, lon)] = [div_id, lat - start_row + ts, lon - start_col + ts] # [division id, relative position in div]

                        div_pos.add(pos)
                        if pos not in neighbors:
                            for dlon,dlat in directions:
                                posLon, posLat = lon + dlon, lat + dlat
                                posNei = n_cols*posLat + posLon
                                if (posLon < start_col or posLon >= end_col) or (posLat < start_row or posLat >= end_row) or (posNei not in valid_locs):
                                    continue
                                neighbors[pos].append(posNei)

            valid_div_pos.append(div_pos)
            div_coords[div_id] = [start_row, end_row, start_col, end_col]
            print("Sub-Region {}: Rows[{} - {}] & Cols[{} - {}] ".format(i*area_ncols+j, start_row, end_row-1, start_col, end_col-1))
    sorted_neighbors = {key: neighbors[key] for key in sorted(neighbors)}
    fname = f'data/map/valid_grids_neighbors_{str(int(grid_size))}m_shanghai_{n}.txt'

    with open(fname, 'w') as file:
        for key, val in sorted_neighbors.items():
            line = str(key) + '|' + ','.join(map(str, val)) + '\n'
            file.write(line)
    file.close()

    return sorted_neighbors, np.array(mat), valid_div_pos, pos_to_div, valid_cells, div_coords

def get_default_mat2D_DQN_padding(mat, ts=10):
    mat_with_padding = np.pad(mat, pad_width=ts, mode='constant', constant_values=invalid_loc_val)
    return mat_with_padding

def get_default_mat2D_DQN(grid_size):

    fhv_locs_visited_cnt_24h_ts, valid_coords = get_default_mat2D_no_padding(grid_size)
    cov_cells = get_visited_grid(grid_size)
    coord_arr = np.array(list(cov_cells))
    fhv_locs_visited_cnt_24h_ts[coord_arr[:, 1], coord_arr[:, 0]] = 0

    return fhv_locs_visited_cnt_24h_ts #, tmp 

def get_default_mat2D_no_padding(grid_size,ts=10):
    valid_cells = get_valid_grid(grid_size)
    visited_cells = get_visited_grid(grid_size)

    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size]

    coordinates_array = np.array(list(valid_cells))

    mat = np.full((n_rows, n_cols), invalid_loc_val, dtype=int)
    mat[coordinates_array[:, 1], coordinates_array[:, 0]] = 1

    return mat, valid_cells

def get_default_mat2D_coverage(grid_size, ts=0):
    valid_cells = set()

    n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[grid_size]
    valid_cells = get_valid_grid(grid_size)

    coordinates_array = np.array(list(valid_cells))

    mat = np.full((n_rows, n_cols), invalid_loc_val, dtype=int)
    mat[coordinates_array[:, 1], coordinates_array[:, 0]] = 1

    return mat, valid_cells

def gen_state(tfm_map, y, x, tf, km_time):
    mat = tfm_map.copy()
    ts = min(tf//km_time, 10)

    mat_padded = get_default_mat2D_DQN_padding(mat)
    y_pad = y+10
    x_pad = x+10

    submatrix = mat_padded[int(y_pad - ts) : int(y_pad + ts + 1), int(x_pad - ts) : int(x_pad + ts + 1)] # For ts matrices
    submatrix = submatrix.copy()
    submatrix[submatrix.shape[0] // 2, submatrix.shape[1] // 2] = dsv_loc_val
    return submatrix

class OrderAssignmentEnv:
    def __init__(self, df, hour, rvs,  visited_grids, name, val_df=None, interval=60, grid_size=500, n_couriers=200, n_rvs=50, rank=1000):
        self.df = df
        self.va_df = val_df
        self.t = 0
        self.hour = hour
        self.interval = interval
        self.n_couriers = n_couriers
        self.n_rvs = n_rvs
        self.rank = rank

        self.valid_grids = get_valid_grid(grid_size)
        self.visited_grids = visited_grids # get_visited_grid(grid_size)
        self.total_cov = len(self.visited_grids)
        self.time_intervals = generate_time_intervals(hour, interval)

        self.orders = []
        self.couriers = self.generate_couriers(self.hour)
        if len(rvs) == 0:
            print(f"No RVs Passed...")
        else:
            self.rvs = rvs

        self.cur_min_LONG = 121.135
        self.cur_max_LONG = 121.8833
        self.cur_min_LAT = 31.0000
        self.cur_max_LAT = 31.3821
        self.block_size = 0.005  # block size, representing (0.5 km)^2

        self.name = name
        self.n = 1
        self.timeframe = 10
        self.grid_size = grid_size
        self.km_time = int(2*grid_size/1000) # km_time to complete 1 km (move to next grid)

        n_rows, n_cols, nPosns, n_rows_valid, n_cols_valid, nPosns_valid = gridSize_info_dict[self.grid_size]
        self.map_area = np.array([[0 for i in range(n_cols_valid)] for j in range(n_rows_valid)])
        self.area_nrows, self.area_ncols = 1, 1
        self.pos_neis, self.tfm_map, self.valid_div_pos, self.pos_to_div, self.valid_coords, self.div_coords = generate_neighbors_list_wBorders(1, grid_size, self.name, self.timeframe, self.km_time)
        self.tfm_maps = []
        self.dsvL = [random.sample(list(self.valid_div_pos[i]), 1)[0] for i in range(self.n)]
        self.dsv_paths = {i:deque([self.dsvL[i]]) for i in range(self.n)}
        self.cov_map = get_default_mat2D_DQN(grid_size)

        self.map_action_to_direction = {0:(-1, -1), 
                                        1:(0, -1), 
                                        2:(1, -1), 
                                        3:(-1, 0), 
                                        4:(1, 0), 
                                        5:(-1, 1), 
                                        6:(0, 1), 
                                        7:(1, 1)}

        self.map_direction_to_action = {(-1, -1):0, 
                                        (0, -1):1, 
                                        (1, -1):2, 
                                        (-1, 0):3, 
                                        (1, 0):4, 
                                        (-1, 1):5, 
                                        (0, 1):6, 
                                        (1, 1):7}
        
        if self.name[:9] == "best_deli":
            self.checkpoint_path = f'models_new/best_{self.name}_model.h5' 
            self.model = models.load_model(self.checkpoint_path) 

        print("\n{} Model Object Created".format(self.name))
        print("\nValid Locations Per Region", [len(self.valid_div_pos[i]) for i in range(self.n)])
        print('\nArea Border of Division (start_row, end_row, start_col, end_col):', self.div_coords)
        print("\nStudy Area Grid Shape: ", self.tfm_map.shape)

    def reset(self):

        self.orders = self.generate_orders(self.time_intervals[self.t][0], self.time_intervals[self.t][1])

    def subreset(self):

        self.orders = self.regenerate_orders()

    def generate_orders(self, start_time, end_time):

        filtered_df = self.df[(self.df['create'] >= start_time) & (self.df['create'] <= end_time)]

        idx = 0
        orders_rem = [order for order in self.orders if order['delivery_time'] <= 0]
        num_orders_rem = len(orders_rem)

        for i in range(len(orders_rem)):
            orders_rem[i]['id'] = i
            orders_rem[i]['deadline'] -= 1 # a duration of 1 minute has passed

        orders = orders_rem
        id_cnt = len(orders)

        for idx, row in filtered_df.iterrows():
            order = {
                'id': id_cnt,
                'pickup': (row['shop_long'], row['shop_lat']),
                'destination': (row['user_long'], row['user_lat']),
                'deadline': row['p_t']//60,
                'fee': row['fee'],
                'delivery_time': 0,
                'courier_id': None
            }
            orders.append(order)
            id_cnt += 1

        return orders

    def regenerate_orders(self):

        idx = 0
        orders_rem = [order for order in self.orders if order['delivery_time'] <= 0 or order['delivery_time'] > order['deadline']]

        for i in range(len(orders_rem)):
            orders_rem[i]['id'] = i
            orders_rem[i]['deadline'] -= 1 # a duration of 1 minute has passed

        orders = orders_rem
        num_orders_rem = len(orders)
        id_cnt = len(orders)

        return orders

    def decrement_order_deadline(self):
        for order in self.orders:
            order['deadline'] -= 1
        orders_rem = [order for order in self.orders if order['delivery_time'] <= 0]
        num_orders_rem = len(orders_rem)
        print(f"Total Orders Remaining After Decrement: {num_orders_rem}")

    def generate_couriers(self, hour, validation=False):

        num_couriers = self.n_couriers
        random.seed(42)
        couriers = []
        for i in range(num_couriers):
            courier = {
                'id': i,
                'location': random.choice(list(self.valid_grids)),
                'pickup_loc': [-1, -1],
                'dest_loc': [-1, -1],
                'incentive_index': [],
                'incentive': [],
                'path_to_dest': deque([]),
                'deadline': None,
                'available_in': 0  # when rider becomes available, could be calculated or updated over time
            }
            couriers.append(courier)
        id_cnt = len(couriers)

        return couriers

    def decrement_courier_availability(self):
        for courier in self.couriers:
            if courier['available_in'] > 0:
                courier['available_in'] = max(0, courier['available_in'] - (self.interval//60))

            if courier['available_in'] == 0:
                if courier['dest_loc'] != [-1, -1]:
                    courier['location'] = courier['dest_loc']
                courier['dest_loc'] = [-1, -1]
                courier['pickup_loc'] = [-1, -1]
                courier['path_to_dest'] = deque([])
                courier['deadline'] = None

        n_available_couriers = len([courier for courier in self.couriers if courier['available_in'] <= 0])

        return n_available_couriers

    def decrement_rv_availability(self):
        for courier in self.rvs:
            if courier['available_in'] > 0:
                courier['available_in'] = max(0, courier['available_in'] - (self.interval//60))
            if not courier['deadline'] == None:
                courier['deadline'] = courier['deadline'] - 1

            if courier['available_in'] == 0:
                if courier['dest_loc'] != [-1, -1]:
                    courier['location'] = courier['dest_loc']
                courier['dest_loc'] = [-1, -1]
                courier['pickup_loc'] = [-1, -1]
                courier['path_to_dest'] = deque([])
                courier['deadline'] = None

        n_available_couriers = len([courier for courier in self.rvs if courier['available_in'] <= 0])

        return n_available_couriers

    def get_state(self, no_action=False):            
        state = []

        for order in self.orders:
            for courier in self.couriers:
                if order['delivery_time'] > 0 or  courier['available_in'] > 0:
                    state.append([-10, -10, -10, -10, -10])
                else:
                    order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
                    order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
                    order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
                    order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
                    courier_long_id, courier_lat_id = courier['location']

                    S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
                    S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
                    deadline_diff = order['deadline'] - (S_p + S_d)
                    courier_path_to_dest = deque(get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id))
                    courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
                    n_new_grids = len(courier_path_to_dest_set - self.visited_grids)

                    state.append([S_p, S_d, deadline_diff, order['fee'], n_new_grids])

        state = np.array(state, dtype=np.float32)

        if len(state) > 0:
            min_vals = np.min(state, axis=0)
            max_vals = np.max(state, axis=0)

            range_vals = max_vals - min_vals

            range_vals = np.where(range_vals == 0, 1, range_vals)

            normalized_state = np.copy(state)
            normalized_state = (state - min_vals) / range_vals
            return np.array(normalized_state, dtype=np.float32)
        else:
            return np.array(state, dtype=np.float32)

    def find_rv_by_id(self, courier_id):
        for courier in self.rvs:
            if courier['id'] == courier_id:
                return courier
        return None

    def find_courier_by_id(self, courier_id):
        for courier in self.couriers:
            if courier['id'] == courier_id:
                return courier
        return None

    def find_order_by_id(self, order_id):
        for order in self.orders:
            if order['id'] == order_id:
                return order
        return None

    def reset_courier_by_id(self, courier_id):
        print(f"Reassiging courier from human courier {courier_id} to RV")
        for courier in self.couriers:
            if courier['id'] == courier_id:
                courier['location'] = courier['location']
                courier['dest_loc'] = [-1, -1]
                courier['pickup_loc'] = [-1, -1]
                courier['path_to_dest'] = deque([])
                courier['deadline'] = None
                courier['available_in'] = 0
        return courier

    def get_valid_actions(self):
        num_orders = len(self.orders)
        num_couriers = len(self.couriers)
        available_couriers = [courier for courier in self.couriers if courier['available_in'] <= 0]
        valid_mask = np.zeros(num_orders * num_couriers, dtype=np.float32)

        for order_id, order in enumerate(self.orders):
            for courier_id, courier in enumerate(self.couriers):
                if courier['available_in'] <= 0 and order['delivery_time'] <= 0:
                    valid_mask[order_id * len(self.couriers) + courier_id] = 1.0

        return valid_mask

    def step(self, action):
        num_orders = len(self.orders)
        num_couriers = len(self.couriers)

        order_id, courier_id = divmod(action, len(self.couriers))
        reward = 0
        done = False
        no_courier_flag = False
        no_order_flag = False

        order = self.orders[order_id]
        courier = self.couriers[courier_id]

        if order['delivery_time'] > 0 or  courier['available_in'] > 0:
            reward = -1e6

        else:               
            order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
            order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
            order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
            order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
            courier_long_id, courier_lat_id = courier['location']

            S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
            S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))

            courier_path_to_dest = deque(get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id))
            delivery_time = len(courier_path_to_dest)
            courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
            n_new_grids = len(courier_path_to_dest_set - self.visited_grids)

            time_saved = order['deadline'] - delivery_time

            if time_saved >= 0:
                sensing_reward = n_new_grids+1
            else:
                sensing_reward = 1/(n_new_grids+1)

            reward = (time_saved*order['fee']*sensing_reward) / (delivery_time+1)

            # Updating Sensing info
            self.visited_grids = self.visited_grids.union(courier_path_to_dest_set)

            if order['courier_id'] != None:
                self.reset_courier_by_id(order['courier_id'])
            order['courier_id'] = courier['id']

            self.couriers[courier_id]['location'] = [courier_long_id, courier_lat_id]

            self.couriers[courier_id]['pickup_loc'] = [order_pickup_long_id, order_pickup_lat_id]
            self.couriers[courier_id]['dest_loc'] = [order_dest_long_id, order_dest_lat_id]
            self.couriers[courier_id]['available_in'] = delivery_time
            self.couriers[courier_id]['path_to_dest'] = deque(courier_path_to_dest)
            self.couriers[courier_id]['deadline'] = order['deadline']

            order['delivery_time'] = len(courier_path_to_dest)

            available_couriers = [courier for courier in self.couriers if courier['available_in']<=0]
            num_available_couriers = len(available_couriers)
            orders_rem = [order for order in self.orders if order['delivery_time'] <= 0]
            num_orders_rem = len(orders_rem)

            if num_available_couriers == 0:
                no_courier_flag = True

            if num_orders_rem == 0:
                no_order_flag = True

        if no_order_flag or no_courier_flag:
            done = True

        next_state = self.get_state()

        return next_state, reward, done, no_courier_flag

    def update(self, order_id, courier_id):
        num_orders = len(self.orders)
        num_couriers = len(self.couriers)

        no_courier_flag = False
        no_order_flag = False

        order = self.find_order_by_id(order_id)

        courier = self.find_courier_by_id(courier_id)

        order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
        order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
        order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
        order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
        courier_long_id, courier_lat_id = courier['location']

        S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
        S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
        delivery_time = S_p + S_d
        courier_path_to_dest = deque(get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id))
        courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
        n_new_grids = len(courier_path_to_dest_set - self.visited_grids)

        time_saved = order['deadline'] - delivery_time        

        if time_saved >= 0:
            sensing_reward = n_new_grids+1
        else:
            sensing_reward = 1/(n_new_grids+1)

        reward = ((time_saved+1e-8)*(order['fee']+1e-8)*sensing_reward) / (delivery_time+1e-8)

        self.visited_grids = self.visited_grids.union(courier_path_to_dest_set)

        order['delivery_time'] = 1
        order['courier_id'] = courier['id']

        courier['pickup_loc'] = [order_pickup_long_id, order_pickup_lat_id]
        courier['dest_loc'] = [order_dest_long_id, order_dest_lat_id]
        courier['available_in'] = delivery_time
        courier['path_to_dest'] = deque(courier_path_to_dest)
        courier['deadline'] = order['deadline']
        courier['incentive_index'].append((order['fee']/(delivery_time+1)))
        courier['incentive'].append((order['fee']))

        available_couriers = [courier for courier in self.couriers if courier['available_in']<=0]
        num_available_couriers = len(available_couriers)
        orders_rem = [order for order in self.orders if order['delivery_time'] <= 0]
        num_orders_rem = len(orders_rem)

        if num_available_couriers == 0:
            no_courier_flag = True

        if num_orders_rem == 0:
            no_order_flag = True

        done = (no_order_flag or no_courier_flag)

        return reward, done

    def update_rvs(self, order_id, courier_id):
        num_orders = len(self.orders)
        num_couriers = len(self.rvs)

        no_courier_flag = False
        no_order_flag = False

        order = self.find_order_by_id(order_id)

        courier = self.find_rv_by_id(courier_id)

        order_pickup_lat_id = int((self.cur_max_LAT - order['pickup'][1]) // self.block_size)
        order_pickup_long_id = int((order['pickup'][0] - self.cur_min_LONG) // self.block_size)
        order_dest_lat_id = int((self.cur_max_LAT - order['destination'][1]) // self.block_size)
        order_dest_long_id = int((order['destination'][0] - self.cur_min_LONG) // self.block_size)
        courier_long_id, courier_lat_id = courier['location']

        S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
        S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
        delivery_time = S_p + S_d
        courier_path_to_dest = deque(get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id))
        courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
        n_new_grids = len(courier_path_to_dest_set - self.visited_grids)

        time_saved = order['deadline'] - delivery_time

        if time_saved >= 0:
            sensing_reward = n_new_grids+1
        else:
            sensing_reward = 1/(n_new_grids+1)

        reward = ((time_saved+1e-8)*sensing_reward)

        order['delivery_time'] = 1
        if order['courier_id'] != None:
            self.reset_courier_by_id(order['courier_id'])
        order['courier_id'] = int(1e6+courier['id'])

        self.rvs[courier_id]['location'] = [courier_long_id, courier_lat_id]

        self.rvs[courier_id]['pickup_loc'] = [order_pickup_long_id, order_pickup_lat_id]
        self.rvs[courier_id]['dest_loc'] = [order_dest_long_id, order_dest_lat_id]

        if order['deadline'] > 0:
            self.rvs[courier_id]['available_in'] = order['deadline']
            self.rvs[courier_id]['deadline'] = order['deadline']
        else:
            self.rvs[courier_id]['available_in'] = delivery_time
            self.rvs[courier_id]['deadline'] = order['deadline']

        order['delivery_time'] = 1

        available_couriers = [courier for courier in self.rvs if courier['available_in']<=0]
        num_available_couriers = len(available_couriers)
        orders_rem = [order for order in self.orders if order['delivery_time'] <= 0]
        num_orders_rem = len(orders_rem)

        if num_available_couriers == 0:
            no_courier_flag = True

        if num_orders_rem == 0:
            no_order_flag = True

        done = (no_order_flag or no_courier_flag)

        return reward, done

    def action_space(self):

        state = self.get_state()
        if np.all(state == 0):
            return -1
        if len(state) > 0:
            return len(state)
        else:
            return -1

    def state_size(self):
        state = self.get_state()
        if np.all(state == 0):
            return -1
        if len(state) > 0:
            return len(state), len(state[0])  # Shape of the state array
        else:
            return -1

    def valid_actions(self, state):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        valid_dirs = [(dx, dy) for dx, dy in n_neighbors if 0 <= y + dy < state.shape[0] and 0 <= x + dx < state.shape[1] and state[y + dy, x + dx] != invalid_loc_val]
        valid_acts = [self.map_direction_to_action[(dx, dy)] for dx, dy in valid_dirs]
        return sorted(valid_acts)

    def choose_action(self, state_features, state):
        valid_acts = self.valid_actions(state)

        q_values = self.model.predict(np.array([state_features.copy()]))
        return valid_acts[np.argmax(q_values[0][valid_acts])]

    def take_action(self, state, action, globalY, globalX):
        next_state = state.copy()

        dirX, dirY = self.map_action_to_direction[action]

        next_globalX, next_globalY = globalX + dirX, globalY + dirY # to update the global study area

        self.cov_map[next_globalY, next_globalX] = 0 # previous dsv location

        return next_globalX, next_globalY

    def get_next_loc_shortest_path_rv(self, vid): # lonStart, latStart, lonPickup, latPickup, lonDest, latDest):
        lonS, latS, lonP, latP, lonD, latD = self.rvs[vid]['location'][0], self.rvs[vid]['location'][1], self.rvs[vid]['pickup_loc'][0], self.rvs[vid]['pickup_loc'][1], self.rvs[vid]['dest_loc'][0], self.rvs[vid]['dest_loc'][1]

        if lonP == -1 and latP == -1:
            dirLong = int((lonD - lonS)/(abs(lonD - lonS))) if abs(lonD - lonS) > 0  else 0
            dirLat = int((latD - latS)/(abs(latD - latS))) if abs(latD - latS) > 0  else 0        
        else:
            dirLong = int((lonP - lonS)/(abs(lonP - lonS))) if abs(lonP - lonS) > 0  else 0
            dirLat = int((latP - latS)/(abs(latP - latS))) if abs(latP - latS) > 0  else 0

        latS += dirLat
        lonS += dirLong
        return lonS, latS

    def rv_sense(self, vid):
        ''' 
        INPUTS:

        OUTPUTS:
        '''    

        courier = self.rvs[vid]
        if courier['location'] == courier['pickup_loc']:
            courier['pickup_loc'] = [-1, -1]
        if courier['location'] == courier['dest_loc']:
            courier['dest_loc'] = [-1, -1]
            courier['available_in'] = 0
            return courier['location']

        delivery_time = get_shortest_path_length_rvs(courier['location'][0], courier['location'][1], courier['pickup_loc'][0], courier['pickup_loc'][1], courier['dest_loc'][0], courier['dest_loc'][1])

        if courier['deadline']:
            sensing_time = courier['deadline'] - delivery_time
        else:
            sensing_time = 10

        if sensing_time <= 0:
            next_lon, next_lat = self.get_next_loc_shortest_path_rv(vid)

        else:
            cur_lon, cur_lat = self.rvs[vid]['location']

            state = gen_state(self.cov_map, cur_lat, cur_lon, sensing_time, 1)

            state_features = self.gen_features_mat(state, sensing_time)

            action = self.choose_action(state_features, state)

            next_lon, next_lat = self.take_action(state, action, cur_lat, cur_lon)

        self.visited_grids.add((next_lon, next_lat))

        if self.rvs[vid]['dest_loc'] != [-1, -1]:
            self.rvs[vid]['path_to_dest'].append([next_lon, next_lat])

        return next_lon, next_lat

    def get_observations(self, state, action, path_rem):
        path_rem = int(min(path_rem, 10))
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        x, y = curX + dirX, curY + dirY

        if not (0 <= y < next_state.shape[0] and 0 <= x < next_state.shape[1]):
            return np.array([-1, -1, -1, -1, -1])

        next_state[curY, curX] = 0 # previous dsv location

        val = next_state[y, x][0]

        cov = val if val >= 0 else -1

        tfm = val if val >= 0 else -1

        n_neighbors = [(dx, dy) for dx in range(-1, 2) for dy in range(-1, 2) if dx != 0 or dy != 0]

        n_nei = sum(1 for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 60)  if val >= 1 else -1 

        nei_tfm = [next_state[y + dy, x + dx] for dx, dy in n_neighbors if 0 <= y + dy < next_state.shape[0] and 0 <= x + dx < next_state.shape[1] and next_state[y + dy, x + dx] >= 0]

        sfm = np.mean(nei_tfm)  if val >= 0 else -1 

        center_x, center_y = x.item(), y.item()
        start_y, end_y =  int(max(center_y - path_rem, 0)), int(min(center_y + path_rem, next_state.shape[0]))
        start_x, end_x =  int(max(center_x - path_rem, 0)), int(min(center_x + path_rem, next_state.shape[1]))
        region = next_state[start_y:end_y+1, start_x:end_x+1]

        rfm = np.mean(region[region >= 0])/60 if val >= 0 else -1

        if self.name == "deliSense":
            return np.array([cov, tfm, n_nei, sfm, rfm])
        elif self.name == "deliSense_woNextLF":
            return np.array([n_nei, sfm, rfm])
        elif self.name == "deliSense_woLocalF":
            return np.array([cov, tfm, rfm])
        elif self.name == "deliSense_woZonalF":
            return np.array([cov, tfm, n_nei, sfm])

    def gen_features_mat(self, state, path_rem):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        features_mat = [self.get_observations(state_copy, action, path_rem) for action in self.map_action_to_direction]
        features_mat = np.array(features_mat)

        min_values = np.min(features_mat, axis=0)
        max_values = np.max(features_mat, axis=0)
        normalized_data = (features_mat - min_values) / (max_values - min_values + 1e-8)

        return normalized_data #features_mat #normalized_data

    def rv_sense_tsmtc(self, vid):
        ''' 
        INPUTS:

        OUTPUTS:
        '''    

        courier = self.rvs[vid]
        if courier['location'] == courier['pickup_loc']:
            courier['pickup_loc'] = [-1, -1]
        if courier['location'] == courier['dest_loc']:
            courier['dest_loc'] = [-1, -1]
            courier['available_in'] = 0
            return courier['location']

        delivery_time = get_shortest_path_length_rvs(courier['location'][0], courier['location'][1], courier['pickup_loc'][0], courier['pickup_loc'][1], courier['dest_loc'][0], courier['dest_loc'][1])

        if courier['deadline']:
            sensing_time = courier['deadline'] - delivery_time
        else:
            sensing_time = 10

        if sensing_time <= 0:
            next_lon, next_lat = self.get_next_loc_shortest_path_rv(vid)

        else:
            cur_lon, cur_lat = self.rvs[vid]['location']

            state = gen_state(self.cov_map, cur_lat, cur_lon, sensing_time, 1)

            state_features = self.gen_features_mat_tsmtc(state, sensing_time)

            action = self.choose_action_baselines(state_features, state)

            next_lon, next_lat = self.take_action(state, action, cur_lat, cur_lon)

        self.visited_grids.add((next_lon, next_lat))

        if self.rvs[vid]['dest_loc'] != [-1, -1]:
            self.rvs[vid]['path_to_dest'].append([next_lon, next_lat])

        return next_lon, next_lat

    def get_observations_tsmtc(self, state, action, path_rem):
        path_rem = int(min(path_rem, 10))
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        x, y = curX + dirX, curY + dirY

        if not (0 <= y < next_state.shape[0] and 0 <= x < next_state.shape[1]):
            return np.array([-1])

        next_state[curY, curX] = 0 # previous dsv location

        val = next_state[y, x][0]

        cov = val if val >= 0 else -1

        return np.array([cov])

    def gen_features_mat_tsmtc(self, state, path_rem):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        features_mat = [self.get_observations_tsmtc(state_copy, action, path_rem) for action in self.map_action_to_direction]
        features_mat = np.array(features_mat)

        min_values = np.min(features_mat, axis=0)
        max_values = np.max(features_mat, axis=0)
        normalized_data = (features_mat - min_values) / (max_values - min_values + 1e-8)

        return normalized_data #features_mat #normalized_data

    def rv_sense_reassign(self, vid):
        ''' 
        INPUTS:

        OUTPUTS:
        '''    

        courier = self.rvs[vid]
        if courier['location'] == courier['pickup_loc']:
            courier['pickup_loc'] = [-1, -1]
        if courier['location'] == courier['dest_loc']:
            courier['dest_loc'] = [-1, -1]
            courier['available_in'] = 0
            return courier['location']

        delivery_time = get_shortest_path_length_rvs(courier['location'][0], courier['location'][1], courier['pickup_loc'][0], courier['pickup_loc'][1], courier['dest_loc'][0], courier['dest_loc'][1])

        if courier['deadline']:
            sensing_time = courier['deadline'] - delivery_time
        else:
            sensing_time = 10

        if sensing_time <= 0:
            next_lon, next_lat = self.get_next_loc_shortest_path_rv(vid)

        else:
            cur_lon, cur_lat = self.rvs[vid]['location']

            state = gen_state(self.cov_map, cur_lat, cur_lon, sensing_time, 1)

            state_features = self.gen_features_mat_reassign(state, sensing_time)

            action = self.choose_action_baselines(state_features, state)

            next_lon, next_lat = self.take_action(state, action, cur_lat, cur_lon)

        self.visited_grids.add((next_lon, next_lat))

        if self.rvs[vid]['dest_loc'] != [-1, -1]:
            self.rvs[vid]['path_to_dest'].append([next_lon, next_lat])

        return next_lon, next_lat

    def get_observations_reassign(self, state, action, path_rem):
        path_rem = int(min(path_rem, 10))
        next_state = state.copy()

        curY, curX = np.where(next_state == dsv_loc_val)

        dirX, dirY = self.map_action_to_direction[action]

        x, y = curX + dirX, curY + dirY

        val = next_state[y, x][0]

        if not (0 <= y < next_state.shape[0] and 0 <= x < next_state.shape[1]):
            return np.array([-1])

        next_state[curY, curX] = 0 # previous dsv location

        center_x, center_y = x.item(), y.item()
        start_y, end_y =  int(max(center_y - path_rem, 0)), int(min(center_y + path_rem, next_state.shape[0]))
        start_x, end_x =  int(max(center_x - path_rem, 0)), int(min(center_x + path_rem, next_state.shape[1]))
        region = next_state[start_y:end_y+1, start_x:end_x+1]

        rfm = np.mean(region[region >= 0])/60 if val >= 0 else -1

        return np.array([rfm])

    def gen_features_mat_reassign(self, state, path_rem):
        state_copy = state.copy()
        y, x = np.where(state_copy == dsv_loc_val)
        features_mat = [self.get_observations_reassign(state_copy, action, path_rem) for action in self.map_action_to_direction]
        features_mat = np.array(features_mat)

        min_values = np.min(features_mat, axis=0)
        max_values = np.max(features_mat, axis=0)
        normalized_data = (features_mat - min_values) / (max_values - min_values + 1e-8)

        return normalized_data #features_mat #normalized_data

    def rv_sense_sdpr(self, vid):
        ''' 
        INPUTS:

        OUTPUTS:
        '''    

        courier = self.rvs[vid]
        if courier['location'] == courier['pickup_loc']:
            courier['pickup_loc'] = [-1, -1]
        if courier['location'] == courier['dest_loc']:
            courier['dest_loc'] = [-1, -1]
            courier['available_in'] = 0
            return courier['location']

        delivery_time = get_shortest_path_length_rvs(courier['location'][0], courier['location'][1], courier['pickup_loc'][0], courier['pickup_loc'][1], courier['dest_loc'][0], courier['dest_loc'][1])

        if courier['deadline']:
            sensing_time = courier['deadline'] - delivery_time
        else:
            sensing_time = 10

        next_lon, next_lat = self.get_next_loc_shortest_path_rv(vid)

        return next_lon, next_lat

    def choose_action_baselines(self, state_features, state):
        valid_acts = self.valid_actions(state)

        return valid_acts[np.argmax(state_features[valid_acts])]
