"""Utility functions to process state."""
from typing import Tuple
import numpy as np
from numba import njit
import shapely.geometry as shp
import heapq

# @jit
# def normalize(array_in):
#     """nx2 or mxnx2"""
#     if len(array_in.shape) == 2:
#         vec, fac = normalize_array(array_in)
#         return vec, fac
#     factors = []
#     vectors = []
#     for m in array_in:
#         vec, fac = normalize_array(m)
#         vectors.append(vec)
#         factors.append(fac)

#     return np.array(vectors), np.array(factors)


@njit
def vector_angles(vecs: np.ndarray) -> np.ndarray:
    """Calculate angles for an array of vectors
    :param vecs: nx2 ndarray
    :return: nx1 ndarray
    """
    ang = np.arctan2(vecs[:, 1], vecs[:, 0])  # atan2(y, x)
    return ang


@njit
def left_normal(vecs: np.ndarray) -> np.ndarray:
    vecs = np.fliplr(vecs) * np.array([-1.0, 1.0])
    return vecs


@njit
def right_normal(vecs: np.ndarray) -> np.ndarray:
    vecs = np.fliplr(vecs) * np.array([1.0, -1.0])
    return vecs


@njit
def normalize(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Normalize nx2 array along the second axis
    input: [n,2] ndarray
    output: (normalized vectors, norm factors)
    """
    norm_factors = []
    for line in vecs:
        norm_factors.append(np.linalg.norm(line))
    norm_factors = np.array(norm_factors)
    normalized = vecs / np.expand_dims(norm_factors, -1)
    # get rid of nans
    for i in range(norm_factors.shape[0]):
        if norm_factors[i] == 0:
            normalized[i] = np.zeros(vecs.shape[1])
    return normalized, norm_factors


@njit
def desired_directions(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given the current state and destination, compute desired direction."""
    destination_vectors = state[:, 4:6] - state[:, 0:2]
    directions, dist = normalize(destination_vectors)
    return directions, dist

@njit
def node_desired_directions(pos: np.ndarray, vgoal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Given the current state and destination, compute desired direction."""
    destination_vectors = vgoal[:, 0:2] - pos[:, 0:2]
    directions, dist = normalize(destination_vectors)
    return directions, dist


@njit
def vec_diff(vecs: np.ndarray) -> np.ndarray:
    """r_ab
    r_ab := r_a ??? r_b.
    """
    diff = np.expand_dims(vecs, 1) - np.expand_dims(vecs, 0)
    return diff


def each_diff(vecs: np.ndarray, keepdims=False) -> np.ndarray:
    """
    :param vecs: nx2 array
    :return: diff with diagonal elements removed
    """
    diff = vec_diff(vecs)
    # diff = diff[np.any(diff, axis=-1), :]  # get rid of zero vectors
    diff = diff[
        ~np.eye(diff.shape[0], dtype=bool), :
    ]  # get rif of diagonal elements in the diff matrix
    if keepdims:
        diff = diff.reshape(vecs.shape[0], -1, vecs.shape[1])

    return diff


@njit
def speeds(state: np.ndarray) -> np.ndarray:
    """Return the speeds corresponding to a given state."""
    #     return np.linalg.norm(state[:, 2:4], axis=-1)
    speed_vecs = state[:, 2:4]
    speeds_array = np.array([np.linalg.norm(s) for s in speed_vecs])
    return speeds_array


@njit
def center_of_mass(vecs: np.ndarray) -> np.ndarray:
    """Center-of-mass of a given group"""
    return np.sum(vecs, axis=0) / vecs.shape[0]


@njit
def minmax(vecs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_min = np.min(vecs[:, 0])
    y_min = np.min(vecs[:, 1])
    x_max = np.max(vecs[:, 0])
    y_max = np.max(vecs[:, 1])
    return (x_min, y_min, x_max, y_max)

@njit
def selecting(state: np.ndarray) -> np.ndarray:
    return [state[:, 7:8] > 0]

#@njit
def perpend(vecs:np.ndarray) -> np.ndarray:
    array = np.stack((vecs[:,1], vecs[:,0]), axis=-1)
    array[:,0] *= -1
    c = array / np.expand_dims(normalize(array)[1], axis=1)
    return c


def find(peds:np.ndarray, rangee:np.ndarray) -> np.ndarray:
    rangee = rangee.reshape((rangee.shape[0], 4, 2))
    boollist = []
    for sect in rangee.tolist():
        booll = []
        for posi in peds.tolist():
            booll.append(shp.Polygon(sect).intersects(shp.Point(posi)))
        boollist.append(booll)
    return np.array(boollist) #y(1D):

@njit
def dense(boool:np.ndarray, area:np.ndarray) -> np.ndarray:
    agents = np.sum(boool, axis=1)
    return agents / area

def render_range(boool:np.ndarray, post:np.ndarray, range_coord:np.ndarray) -> dict:
    start_point = {}
    for peo in range(post.shape[0]):
        a = boool[:, peo] 
        if range_coord[a].size > 4:
            copy = np.copy(range_coord[a])
            #print(copy)
            multi_pos = copy.reshape((copy.shape[0], 2, 2))
            x0y0 = multi_pos[0, 0]
            #print(multi_pos)
            #print(np.repeat(np.array([[x0y0],[x0y0]]), copy.size/4, axis=1).reshape(copy.shape[0], 2, 2) == multi_pos)
            booll = np.repeat(np.array([[x0y0],[x0y0]]), copy.size/4, axis=1).reshape(copy.shape[0], 2, 2) == multi_pos
            b = 0
            for _ in booll:
                c = 0
                for __ in _:
                    if __.sum() == 2:
                        c += 1
                b += c
            if b == copy.shape[0]:#-->first
                start_point[str(peo)] = x0y0
            else:#-->second
                start_point[str(peo)] = multi_pos[0, 1]
        else:
            copy = range_coord[a]
            start_point[str(peo)] = [copy[0,0:2],copy[0,2:4]]
    #print(start_point)
    return start_point
def weight_single(points:list, target:np.ndarray, area:np.ndarray, coord:np.ndarray, length:np.ndarray, peds:np.ndarray, r_vec:np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p1, p2 = points
    coord_p = np.concatenate(points, axis=None)
    boll = coord == coord_p
    ans = []
    for _ in boll:
        ans.append(_.sum() == 4)
    #print(ans)
    #print(r_vec[0])
    #print(points)
    ####find point on the line
    n1 = normalize(np.expand_dims(p1 - target, axis=0))[1]
    n2 = normalize(np.expand_dims(p2 - target, axis=0))[1]
    nt = n1 + n2
    #print(np.expand_dims(p2 - target, axis=0))
    mid = p1*(n1)/nt + p2*(n2)/nt
    #print(mid)
    #print((p1 + r_vec[ans]).tolist(), (p1 - r_vec[ans]).tolist(), (mid - r_vec[ans]).tolist(), (mid + r_vec[ans]).tolist())
    ###
    ###calc range & dense
    rangee1 = np.expand_dims(np.concatenate([(p1 + r_vec[ans]).tolist()[0], (mid + r_vec[ans]).tolist()[0], (mid - r_vec[ans]).tolist()[0], (p1 - r_vec[ans]).tolist()[0]],axis=None),axis=0)
    rangee2 = np.expand_dims(np.concatenate([(p2 + r_vec[ans]).tolist()[0], (mid + r_vec[ans]).tolist()[0], (mid - r_vec[ans]).tolist()[0], (p2 - r_vec[ans]).tolist()[0]],axis=None),axis=0)
    #print(area[ans])
    #print(rangee1)
    dense1 = dense(find(peds, rangee1), area[ans]*(n1)/nt)
    dense2 = dense(find(peds, rangee2), area[ans]*(n2)/nt)
    #print(dense1)
    #print(dense2)
    ###
    l1 = length[ans]*n1/nt
    l2 = length[ans]*n2/nt
    #find(peds, )
    #print(rangee2)
    return dense1 , l1 , dense2 , l2

def dijkstra(graph, s, finish):
    dists = {node: float('inf') for node in graph}
    dists[s] = 0
    priority_queue = [(0, s)]
    pre = {}
    while priority_queue:
        curr_dist, curr_node = heapq.heappop(priority_queue)
        if curr_node == finish:
            path = []
            while curr_node in pre:
                path.insert(0, curr_node)
                curr_node = pre[curr_node]
            path.insert(0, s)
            return dists[finish], path
        if curr_dist > dists[curr_node]:
            continue
        for neighbor, weight in graph[curr_node].items():
            dist = curr_dist + weight
            if dist < dists[neighbor]:
                dists[neighbor] = dist
                pre[neighbor] = curr_node
                heapq.heappush(priority_queue, (dist, neighbor))
    return None, None



#@njit
#def node_desired_directions(self,  state: np.ndarray):
#    
#@njit
#def dijkstra():
#    
