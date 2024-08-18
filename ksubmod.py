#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import concurrent.futures
import os
# TO USE SPECIFIED NUMBER OF THREADS *nThreads* and not parallelize internally by python
os.environ["MKL_NUM_THREADS"] = "1" 
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys

from datetime import datetime
import numpy as np
import random
from queue import PriorityQueue
# from scipy import sparse
from scipy.sparse import lil_matrix, csr_matrix
import pandas as pd

try:
    from mpi4py import MPI
except ImportError:
    MPI = None 
if not MPI:
    print("MPI not loaded from the mpi4py package. Serial implementations will function, \
            but parallel implementations will not function.")
import time

from envs import get_visited_grid, get_shortest_path, get_shortest_path_length_rvs, generate_time_intervals, get_valid_grid, generate_rvs, OrderAssignmentEnv

def sample_seq( X, k, randstate ):
    if len(X) <= k:
        randstate.shuffle(X)
        return X
    Y = list(randstate.choice(X, k, replace=False));
    randstate.shuffle(Y);
    return Y;

hourly_riders_dict = {0: 118, 1: 65, 2: 48, 3: 15, 4: 23, 5: 17, 6: 55, 7: 211, 8: 441, 9: 622, 10: 1547, 11: 2192, 12: 1764, 13: 1233, 14: 837, 15: 750, 16: 829, 17: 1104, 18: 1091, 19: 834, 20: 589, 21: 354, 22: 236, 23: 130}
gridSize_info_dict = {1000:(38, 74, 2812, 38, 74, 2812), 500:(76, 148, 11248, 76, 148, 11248)}
hourly_orders_dict = {0: 1581, 1: 988, 2: 541, 3: 309, 4: 213, 5: 276, 6: 1059, 7: 2741, 8: 6147, 9: 6806, 10: 22318, 11: 59805, 12: 29838, 13: 15980, 14: 11526, 15: 9451, 16: 11264, 17: 19842, 18: 20461, 19: 14938, 20: 9925, 21: 5542, 22: 4025, 23: 1770}

cur_min_LONG = 121.135
cur_max_LONG = 121.8833
cur_min_LAT = 31.0000
cur_max_LAT = 31.3821
block_size = 0.005

'''
Utility functions
'''

class KSubmodularDispatch:  

    def __init__(self, orders, couriers, visited_grids, rv=False, rank=0):

        self.orders     = orders
        self.couriers    = couriers
        self.orders_f   = set()
        self.couriers_f  = set()
        self.od_f       = set()
        self.sparse = False
        self.sensing_cov  = visited_grids
        self.rv = rv

        self.cur_min_LONG = 121.135
        self.cur_max_LONG = 121.8833
        self.cur_min_LAT = 31.0000
        self.cur_max_LAT = 31.3821
        self.block_size = 0.005  # block side length, representing 0.5 km
        self.rank = rank

        self.delivery_quotient_mat, self.sensing_quotient_mat = self.gen_quotient()
        if rank ==0:
            print(f"self.delivery_quotient_mat.shape: {self.delivery_quotient_mat.shape};  self.sensing_quotient_mat.shape: {self.sensing_quotient_mat.shape};  len(self.sensing_cov): {len(self.sensing_cov)}")

    def reset(self):
        self.orders_f     = set()

        self.delivery_quotient_mat, self.sensing_quotient_mat = self.update_quotient()

    def gen_quotient_sparse(self):
        delivery_quotient_mat = lil_matrix(len(self.orders), len(self.couriers))
        sensing_quotient_mat = lil_matrix(len(self.orders), len(self.couriers))

        for i, order in enumerate(self.orders):
            for j, courier in enumerate(self.couriers):
                if (i, j) in self.od_f:
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

                if not self.rv:
                    delivery_quotient = ((time_saved+1e-8)*(order['fee']+1e-8)) / (delivery_time+1e-8)
                else:
                    delivery_quotient = (time_saved+1e-8)
                delivery_quotient_mat[i, j] = delivery_quotient

                courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
                n_new_grids = len(courier_path_to_dest_set - self.sensing_cov)

                if time_saved >= 0:
                    sensing_reward = n_new_grids+1
                else:
                    sensing_reward = 1/(n_new_grids+1)

                sensing_quotient_mat[i, j] = sensing_reward

        return delivery_quotient_mat, sensing_quotient_mat

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

                if not self.rv:
                    delivery_quotient = ((time_saved+1e-8)*(order['fee']+1e-8)) / (delivery_time+1e-8)
                else:
                    delivery_quotient = (time_saved+1e-8)
                delivery_quotient_mat[i].append(delivery_quotient)

                courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
                n_new_grids = len(courier_path_to_dest_set - self.sensing_cov)

                if time_saved >= 0:
                    sensing_reward = n_new_grids+1
                else:
                    sensing_reward = 1/(n_new_grids+1)

                sensing_quotient_mat[i].append(sensing_reward)

        return np.array(delivery_quotient_mat), np.array(sensing_quotient_mat)

    def get_quotient_mat(self):
        dispatch_quotient = self.delivery_quotient_mat*self.sensing_quotient_mat
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

                if not self.rv:
                    delivery_quotient = ((time_saved+1e-8)*(order['fee']+1e-8)) / (delivery_time+1e-8)
                else:
                    delivery_quotient = (time_saved+1e-8)
                delivery_quotient_mat[i].append(delivery_quotient)

                courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
                n_new_grids = len(courier_path_to_dest_set - self.sensing_cov)

                if time_saved >= 0:
                    sensing_reward = n_new_grids+1
                else:
                    sensing_reward = 1/(n_new_grids+1)

                sensing_quotient_mat[i].append(sensing_reward)

        return np.array(delivery_quotient_mat), np.array(sensing_quotient_mat)

    def get_best_dispatch(self):
        dispatch_quotient = self.delivery_quotient_mat*self.sensing_quotient_mat

        action, reward = np.argmax(dispatch_quotient), np.max(dispatch_quotient)
        order_id, courier_id = divmod(action, len(self.couriers))

        return self.orders[order_id]['id'], self.couriers[courier_id]['id'], reward

    def threshold_dispatch(self, tau=0):
        dispatch_quotient = self.delivery_quotient_mat*self.sensing_quotient_mat

        action, reward = np.argmax(dispatch_quotient), np.max(dispatch_quotient)
        if reward >= tau:
            order_id, courier_id = divmod(action, len(self.couriers))

            return self.assign(order_id, courier_id)
        else:
            if reward == -1e6:
                return -2, -2, -2, -2
            else:
                return -1, -1, -1, -1

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

def fetch_info(o_id, d_id, orders, couriers):
    for i, o in enumerate(orders):
        if o['id'] == o_id:
            order = o
            del orders[i]
            break
    for j, d in enumerate(couriers):
        if d['id'] == d_id:
            courier = d
            del couriers[j]
            break
    return order, courier, orders, couriers

def get_sensing_cov(order, courier, sensing_cov):
    order_pickup_lat_id = int((cur_max_LAT - order['pickup'][1]) // block_size)
    order_pickup_long_id = int((order['pickup'][0] - cur_min_LONG) // block_size)
    order_dest_lat_id = int((cur_max_LAT - order['destination'][1]) // block_size)
    order_dest_long_id = int((order['destination'][0] - cur_min_LONG) // block_size)
    courier_long_id, courier_lat_id = courier['location']

    courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)
    courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
    sensing_cov = courier_path_to_dest_set | sensing_cov
    return sensing_cov

def gen_quotient_one(order, courier, rv, sensing_cov):

    order_pickup_lat_id = int((cur_max_LAT - order['pickup'][1]) // block_size)
    order_pickup_long_id = int((order['pickup'][0] - cur_min_LONG) // block_size)
    order_dest_lat_id = int((cur_max_LAT - order['destination'][1]) // block_size)
    order_dest_long_id = int((order['destination'][0] - cur_min_LONG) // block_size)
    courier_long_id, courier_lat_id = courier['location']

    S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
    S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
    delivery_time = S_p + S_d
    time_saved = order['deadline'] - (S_p + S_d)
    courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)

    if not rv:
        delivery_quotient = ((time_saved+1e-8)*(order['fee']+1e-8)) / (delivery_time+1e-8)
    else:
        delivery_quotient = (time_saved+1e-8)

    courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
    n_new_grids = len(courier_path_to_dest_set - sensing_cov)

    if time_saved >= 0:
        sensing_reward = n_new_grids+1
    else:
        sensing_reward = 1/(n_new_grids+1)
    return sensing_reward*delivery_quotient

def gen_quotient_all(orders, couriers, sensing_cov, rv):

    quotient_mat = []
    for i, order in enumerate(orders):
        for j, courier in enumerate(couriers):
            quotient = gen_quotient_one(order, courier, rv, sensing_cov)
            quotient_mat.append([quotient, order['id'], courier['id'], i, j])

    return quotient_mat

def K_Submodular_TopN_OrderDispatch(orders, couriers, k, visited_grids, rv, comm, rank, size, p_root=0):
    ''' 
    Accelerated lazy algorithm for obtaining the Top ranked couriers for each of the assigned orders.
    **NOTE** solution sets and values may be different than those found by our implementation, 
    as the two implementations may break ties differently (i.e. when two elements 
    have the same marginal value,  the two implementations may not pick the same element to add)

    INPUTS:
    int k -- the cardinality constraint
    
    OUTPUTS:
    list L -- the solution, where each element in the list is an element in the solution set.
    '''
    objective = KSubmodularDispatch(orders.copy(), couriers.copy(), visited_grids.copy(), rv, rank)

    comm.barrier()

    D = []
    if len(orders) == 0:
        return D
    couriers_assigned = set()

    if len(couriers) == 0:
        return []
    for i in range(k):

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

def Distributed_TopN(orders, couriers, k, visited_grids,rv, comm, rank, size, p_root=0, seed=42, nthreads=16):

    comm.barrier()
    order_split_local = orders[rank] # np.array_split(V, size)[rank]

    ele_A_local_dispatches = [ K_Submodular_TopN_OrderDispatch(order_split_local, couriers, k, visited_grids, rv, comm, rank, size, p_root=0)]

    ele_dispatches = comm.allgather(ele_A_local_dispatches)

    return [d for sublist in ele_dispatches for d in sublist]


def K_Submodular_Threshold_Dispatch(orders, couriers, visited_grids, rv, comm, rank, size, p_root=0, seed=42, epsilon=0.01):

    """
    Implements the k-submodular Threshold Dispatch algorithm.

    Returns:
    set: An item-index pair set S.
    """

    k = len(orders)
    if not k :
        return []
    S = set()
    B = len(orders)

    D = []
    od_f = set()
    d_f = set()
    o_f = set()
    sensing_cov = visited_grids
    max_reward = -1e6

    for i, order in enumerate(orders):
        for j, courier in enumerate(couriers):
            if (i, j) in od_f or j in d_f or i in o_f:
                continue
            order_pickup_lat_id = int((cur_max_LAT - order['pickup'][1]) // block_size)
            order_pickup_long_id = int((order['pickup'][0] - cur_min_LONG) // block_size)
            order_dest_lat_id = int((cur_max_LAT - order['destination'][1]) // block_size)
            order_dest_long_id = int((order['destination'][0] - cur_min_LONG) // block_size)
            courier_long_id, courier_lat_id = courier['location']

            S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
            S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
            delivery_time = S_p + S_d
            time_saved = order['deadline'] - (S_p + S_d)
            courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)

            courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
            n_new_grids = len(courier_path_to_dest_set - sensing_cov)

            if time_saved < 0:

                    time_saved = (1/(1-time_saved))
            else:
                time_saved += 1

            sensing_quotient = n_new_grids+1
            if not rv:
                delivery_quotient = ((time_saved+1e-8)*(order['fee']+1e-8)) / (delivery_time+1e-8)
            else:
                delivery_quotient = (time_saved+1e-8)

            max_reward = max(max_reward, delivery_quotient*sensing_quotient)

    d = max_reward
    if d == -1e6:
        return []
    tau = d
    min_tau = (1 - epsilon) * epsilon * d / (3 * B)

    while tau > min_tau:
        if len(o_f) == len(orders):
            break
        if len(d_f) == len(couriers):
            break
        for i, order in enumerate(orders):
            for j, courier in enumerate(couriers):
                if (i in o_f or (i, j) in od_f):
                    break
                if (j in d_f):
                    continue
                order_pickup_lat_id = int((cur_max_LAT - order['pickup'][1]) // block_size)
                order_pickup_long_id = int((order['pickup'][0] - cur_min_LONG) // block_size)
                order_dest_lat_id = int((cur_max_LAT - order['destination'][1]) // block_size)
                order_dest_long_id = int((order['destination'][0] - cur_min_LONG) // block_size)
                courier_long_id, courier_lat_id = courier['location']

                S_p = max(abs(order_pickup_lat_id - courier_lat_id) , abs(order_pickup_long_id - courier_long_id))
                S_d = max(abs(order_pickup_lat_id - order_dest_lat_id) , abs(order_pickup_long_id - order_dest_long_id))
                delivery_time = S_p + S_d
                time_saved = order['deadline'] - (S_p + S_d)
                courier_path_to_dest = get_shortest_path(courier_long_id, courier_lat_id, order_pickup_long_id, order_pickup_lat_id, order_dest_long_id, order_dest_lat_id)

                courier_path_to_dest_set = set([(x,y) for x, y in courier_path_to_dest])
                n_new_grids = len(courier_path_to_dest_set - sensing_cov)

                if time_saved < 0:

                    time_saved = (1/(1-time_saved))
                else:
                    time_saved += 1

                sensing_quotient = n_new_grids+1
                if not rv:
                    delivery_quotient = ((time_saved+1e-8)*(order['fee']+1e-8)) / (delivery_time+1e-8)
                else:
                    delivery_quotient = (time_saved+1e-8)

                dispatch_quotient = delivery_quotient*sensing_quotient

                if dispatch_quotient >= tau:
                    o_f.add(i)
                    d_f.add(j)
                    od_f.add((i, j))
                    sensing_cov = sensing_cov.union(courier_path_to_dest_set)

                    D.append([dispatch_quotient, order['id'], courier['id']])
                    break

        tau *= (1 - epsilon)

    return D

def Dist_Submodular_OrderDispatch(orders, couriers, visited_grids, rv, comm, rank, size, p_root=0, seed=42):
    """
    Implements the k-submodular Threshold Dispatch algorithm.

    Returns:
    set: An item-index pair set S.
    """

    comm.barrier()
    order_split_local = np.array_split(orders, size)[rank]
    # if rank == 0:

    ele_A_local_dispatches = gen_quotient_all(order_split_local, couriers, visited_grids, rv)

    ele_dispatches = comm.allgather(ele_A_local_dispatches)

    return [d for sublist in ele_dispatches for d in sublist]

    return D

def DS_MR_OD(orders, couriers, k, eps, visited_grids, rv, comm, rank, size, p_root=0, seed=42, nthreads=16):
    '''
    The parallelizable distributed algorithm DeliSense-MapReduce_OrderDispatch. Uses multiple machines to obtain solution
    PARALLEL IMPLEMENTATION (Multithread)

    
    OUTPUTS:
    list S -- the solution, where each pair is a order-courier matching.
    ''' 

    comm.barrier()
    p_start = MPI.Wtime()

    S = []
    time_rounds = [0]
    query_rounds = [0]
    queries = 0
    random.seed(seed)

    q = np.random.RandomState(42)

    V = [[] for i in range(size)]

    for o in range(len(orders)):
        x = random.randint(0, size-1)
        V[x].append(orders[o])

    p_start_dist = MPI.Wtime()

    if rank == 0:
        print("\nStarting Distributed K_Submodular_TopN_OrderDispatch...")
    S_DistGreedy_split = Distributed_TopN(V, couriers, k, visited_grids, rv, comm, rank, size, p_root, seed)
    if rank == 0:
        print("\nCompleted Distributed K_Submodular_TopN_OrderDispatch\n")

    S_DistGreedy = []
    S_DistGreedy_all = [] 
    for i in range(len(S_DistGreedy_split)):
        S_DistGreedy.extend(list(S_DistGreedy_split[i]))
        S_DistGreedy_all.append(list(S_DistGreedy_split[i]))
    S_DistGreedy = list(np.unique(S_DistGreedy))
    p_stop_dist = MPI.Wtime()

    time_dist = (p_stop_dist - p_start_dist)

    p_start_post = MPI.Wtime()
    couriers_subset = set()
    couriers_updated = []

    merged_S_Dist = [item for sublist in S_DistGreedy_all for item in sublist]

    for i, (o, d, o_id, d_id) in enumerate(merged_S_Dist):
        if d not in couriers_subset:
            couriers_updated.append(couriers[d])
            couriers_subset.add(d)

    D = K_Submodular_Threshold_Dispatch(orders, couriers_updated, visited_grids, rv, comm, rank, size, p_root=0, seed=42)

    if rank == 0:
        print("\nCompleted K_Submodular_Threshold_Dispatch\n")

    return D

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    n_rvs = int(sys.argv[1]) # Number of AVs
    k = 2
    out_fname=f"results/MR-KSubMod-Result_DeliSense_{n_rvs}.csv"
    if rank == 0:
        print('Initializing run')

    rvs = generate_rvs(n_rvs)
    visited_grids = get_visited_grid(500)
    interval = 60

    df = pd.read_csv('data/order_data_test_2020_1013-1014_zhonghuan.csv')
    df['create'] = pd.to_datetime(df['create'])

    results = []

    env = OrderAssignmentEnv(df, 0, rvs, visited_grids, interval=interval, n_couriers=hourly_riders_dict[0], n_rvs=n_rvs, rank=rank)
    overdue_cnt = 0

    for h in range(24):
        if rank==0:
            print("Initiating New OrderAssignmentEnv")
        env = OrderAssignmentEnv(df, 0, rvs, visited_grids, interval=interval, n_couriers=hourly_riders_dict[h], n_rvs=n_rvs, rank=rank)

        env.t = 0
        env.time_intervals = generate_time_intervals(h, interval)

        start_time = time.time()
        couriers = []
        for i in range(int(60*(60/env.interval))):
            if rank == 0:
                print(f"\n<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>\n")

            env.reset()
            orders_rem = [order for order in env.orders if order['delivery_time'] <= 0]
            orders_rem_old = orders_rem
            if rank == 0:
                print(f"Rank : {rank}>> Orders Remaining At Start: {len(orders_rem)}")

            couriers = [courier for courier in env.couriers if courier['available_in'] <= 0]

            if rank == 0:
                print(f"Rank : {rank}>> Available Drivers Before Decrement: {len(couriers)}")
            n_courier = env.decrement_courier_availability()

            couriers_rem = [courier for courier in env.couriers if courier['available_in'] <= 0]
            if rank == 0:
                print(f"Rank : {rank}>> Available Drivers After Decrement: {len(couriers_rem)}")
            while len(orders_rem) != 0:    
                orders_rem = [order for order in env.orders if order['delivery_time'] <= 0]
                if rank == 0:
                    print(f"RANK: {rank}>>({env.time_intervals[env.t][0]} - {env.time_intervals[env.t][1]}) >> Generated Orders: {len(orders_rem)};   Previous # Orders: {len(orders_rem_old)}") 
                orders_rem_old = orders_rem
                if len(orders_rem) == 0:
                    if rank == 0:
                        print("No Order Remaining For Human...")
                    break
                if len(couriers_rem) == 0:
                    if rank == 0:
                        print("No Driver Available")
                    break
                D = DS_MR_OD(orders_rem, couriers_rem, k, 0.05, visited_grids, False, comm, rank, size, p_root=0, seed=42)
                total_rew = 0
                orders_rem = [order for order in env.orders if order['delivery_time'] <= 0]

                for rew, o_id, d_id in D:
                    reward, done = env.update(o_id, d_id)

                    orders_rem = [order for order in env.orders if order['delivery_time'] <= 0]
                    couriers_TMP = [courier for courier in env.couriers if courier['available_in'] <= 0]
                    total_rew += rew
                    if reward < 0:
                        overdue_cnt += 1

                if rank == 0:
                    orders_rem = [order for order in env.orders if order['delivery_time'] <= 0]
                    couriers_rem = [courier for courier in env.couriers if courier['available_in'] <= 0]

                visited_grids = env.visited_grids

                orders_rem = [order for order in env.orders if order['delivery_time'] <= 0]

                couriers_rem = [courier for courier in env.couriers if courier['available_in'] <= 0]

            if rank ==0:
                print(f"\nHUMAN ASSIGNMENTS COMPLETE\nSTARTING AV ASSIGNMENTS\n")

            # AV Assignments
            env.subreset()
            orders_rem = [order for order in env.orders if order['delivery_time'] <= 0 or (order['delivery_time'] > order['deadline'] and order['courier_id'] < 1e6)]
            orders_rem_old = orders_rem

            couriers = [courier for courier in env.rvs if courier['available_in'] <= 0]
            env.decrement_rv_availability()
            couriers_rem = [courier for courier in env.rvs if courier['available_in'] <= 0]
            while len(orders_rem) != 0:

                orders_rem = [order for order in env.orders if order['delivery_time'] <= 0 or (order['delivery_time'] > order['deadline'] and order['courier_id'] < 1e6)]
                orders_rem_old = orders_rem

                if len(orders_rem) == 0:
                    if rank == 0:
                        print("\nNo Order Remaining For AVs...")
                    break

                couriers = []
                couriers_rem = [courier for courier in env.rvs if courier['available_in'] <= 0]
                if rank ==0:
                    print(f"Currently Available AVs: {len(couriers_rem)}")
                if len(couriers_rem) == 0:
                    if rank == 0:
                        print("No AV Available")
                    break

                # AV Driver Assignments
                D = DS_MR_OD(orders_rem, couriers_rem, k, 0.05, visited_grids, True, comm, rank, size, p_root=0, seed=42)
                total_rew = 0

                for rew, o_id, d_id in D:
                    reward, done = env.update_rvs(o_id, d_id)
                    total_rew += reward
                if rank == 0:
                    print(f"\nCompleted AV Assignments >> Time({env.time_intervals[env.t][0]} - {env.time_intervals[env.t][1]}) (Assigned {len(D)} Orders) (Total Reward: {total_rew})\n")
                rvs = env.rvs
                visited_grids = env.visited_grids
                orders_rem = [order for order in env.orders if order['delivery_time'] <= 0]
                couriers_rem = [courier for courier in env.rvs if courier['available_in'] <= 0]

            orders_overdue = [order for order in env.orders if order['delivery_time'] > order['deadline']]
            overdue_cnt += len(orders_overdue)

            for vid in range(len(env.rvs)):
                env.rv_sense(vid)

            env.t += 1    

        end_time = time.time()
        execution_time = end_time - start_time

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

        if rank == 0:
            print(f"\n-x-x-x-x-x-HOURLY RESULT-x-x-x-x-x-\n")
            print(f"Hour ({h}) >> Total Sensing Coverage: {len(visited_grids)}")
            print(f"Hour ({h}) >> New Locations Sensing Coverage: {new_locs}")
            print(f"Hour ({h}) >> Order Overdue: {overdue_cnt}")
            print(f"Hour ({h}) >> Average Driving Incentive : {avg_di}")   
            print(f"Hour ({h}) >> Average Driving Incentive Index: {avg_dii}")   
            print(f"\n-x-x-x-x-x-x-x-x-x-x-\n")

            results.append({
                "Model": "DeliSense",
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
            res_df.to_csv(out_fname)

        overdue_cnt = 0
        visited_grids = get_visited_grid(500)

    if rank == 0:
        res_df = pd.DataFrame(results)
        res_df.to_csv(out_fname)
    print("Run Complete")

