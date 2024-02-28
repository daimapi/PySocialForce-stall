"""PedState and EnvState"""

"""This module tracks the state odf scene and scen elements like pedestrians, groups and obstacles"""
from typing import List

import numpy as np

from pysocialforce.utils import stateutils, logger


class PedState:
    """Tracks the state of pedstrains and social groups"""

    def __init__(self, state, goals, groups, config):
        logger.info("Pedstate init")
        self.default_tau = config("tau", 0.5) #dude
        self.step_width = config("step_width", 0.4)
        self.agent_radius = config("agent_radius", 0.35)
        self.max_speed_multiplier = config("max_speed_multiplier", 1.3)

        self.max_speeds = None
        self.initial_speeds = None

        self.ped_states = []
        self.group_states = []
        self.goals = goals
        
        #state = np.concatenate((state, np.reshape(goals[:, 0], (state.shape[0], 3))), axis = -1)
        state = np.insert(state, 4, state[:, 0], axis=1)
        state = np.insert(state, 5, state[:, 1], axis=1)
        state = np.concatenate((state, np.expand_dims(np.zeros(state.shape[0]), -1)), axis=-1) #add goaln
        logger.info("among us")
        logger.info(state)
        #input  [px, py, vx, vy, pt]
        #mask1  [px, py]
        #append [px, py, vx, vy, px, py, pt]
        #goaln  [px, py, vx, vy, px, py, pt, 0]
        #output [px, py, vx, vy, px, py, pt, 0, tau]
        #[px, py, vx, vy, g1x, g1y, g1t, gn, tau]
        self.update(state, groups)

    def update(self, state, groups):
        logger.info("Pedstate updating")
        self.state = state
        self.groups = groups

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state):
        # when someone type self(pedstate).state=...
        tau = self.default_tau * np.ones(state.shape[0]) #tau = array([deftau,deftau,deftau...,deftau]) amount of deftaus depends on the number of agents (1stDim of state(ped init state list))
        #for n in range(state.shape[0]):
        #    self._state[n] = np.append(state[n], goal[n, 0])
        if state.shape[1] < 9: #num of elements in each agents < 10
            self._state = np.concatenate((state, np.expand_dims(tau, -1)), axis=-1) #add deftau in back of each agents (no.7)
        else:
            self._state = state

        if self.initial_speeds is None:
            self.initial_speeds = self.speeds()
        self.max_speeds = self.max_speed_multiplier * self.initial_speeds
        self.ped_states.append(self._state.copy())
        logger.debug(f"self._state.copy():\n {repr(self._state.copy())}")

    def get_states(self):
        return np.stack(self.ped_states), self.group_states
    
    def get_pure_state(self):
        return np.stack(self.ped_states)

    def size(self) -> int:
        return self.state.shape[0]

    def pos(self) -> np.ndarray:
        return self.state[:, 0:2]

    def vel(self) -> np.ndarray:
        return self.state[:, 2:4]

    def goal(self) -> np.ndarray:
        return self.state[:, 4:6]
    
    def t(self) -> np.ndarray:
        return self.state[:, 6]#np.concatenate(self.state[:, 6:7], axis=None) #1d
    
    def goaln(self, n:int) -> np.ndarray:
        n = n - 1
        return self.goals[:, n, 0:2]
    
    def goaltn(self, n:int) -> np.ndarray:
        n = n - 1
        return self.goals[:, n, 2:3]
    
    def num(self) -> np.ndarray:
        return self.state[:, 7:8] #2d
    
    def numlenght(self) -> int:
        return self.goals.shape[1]

    def tau(self):
        return self.state[:, 8:9]

    def speeds(self):
        """Return the speeds corresponding to a given state."""
        return stateutils.speeds(self.state)

    def step(self, force, groups=None):
        """Move peds according to forces"""
        logger.info("move peds a step by (the methond of pedstate).step")
        ###desired velocity
        desired_velocity = self.vel() + self.step_width * force
        desired_velocity = self.capped_velocity(desired_velocity, self.max_speeds)
        ###stop when arrived
        #logger.debug("unmod desired vel :")
        #logger.debug(desired_velocity)
        
        arrivedf_mask = np.logical_and(stateutils.desired_directions(self.state)[1] < 0.5 , self.t() == 0) #may fix turbo
        arrived_mask = np.logical_and(stateutils.desired_directions(self.state)[1] < 0.5 , self.t() > 0)
        #logger.debug("agents dist tar")
        #logger.debug(stateutils.desired_directions(self.state)[1])
        #logger.debug("t:")
        #logger.debug(self.t())
        #print(arrived_mask)
        #print(self.state[:, 6:7][arrived_mask])
        #print(np.expand_dims(np.ones(self.size())[arrived_mask], axis=1))
        #print(np.shape(self.state[:, 6:7][arrived_mask]))
        desired_velocity[stateutils.desired_directions(self.state)[1] < 0.5] = [0, 0]
        #if (np.shape(self.state[:, 6:7][arrived_mask]) != (0,1)):
        #self.state[:, 6:7][arrived_mask] = np.subtract(
        #    self.state[:, 6:7][arrived_mask], np.expand_dims(
        #        np.ones(
        #            self.size()
        #        )[arrived_mask], axis=1
        #    )
        #)
        self.t()[arrived_mask] += -1
        #logger.debug("moded desired vel :")
        #logger.debug(desired_velocity)
        a = 0
        for n in self.num()[arrivedf_mask] :
            n = int(n.tolist()[0])
            if n <= self.numlenght() and n >= 0 :
                self.num()[np.where(arrivedf_mask)[0][a]] += 1
                self.goal()[np.where(arrivedf_mask)[0][a]] = self.goaln(n)[arrivedf_mask][a] ###dont (need) to change pos
                self.t()[np.where(arrivedf_mask)[0][a]] = self.goaltn(n)[arrivedf_mask][a]

            else:
                self.num()[np.where(arrivedf_mask)[0][a]] = -1 #dude
            a += 1
        
        

        #if (stateutils.desired_directions(self.state)[1] < 0.5) : get fucked
        #    self.state[:, 0:2] = self.state[:, 4:6] ###dont change this
        #    self.state[:, 4:6] = self.state[:, 6:8]
        ###update state
        next_state = self.state
        #logger.debug("self_state-pos:")
        #logger.debug(next_state[:, 0:2])
        next_state[:, 0:2] += desired_velocity * self.step_width
        #logger.debug("next_state-pos:")
        #logger.debug(next_state[:, 0:2])
        next_state[:, 2:4] = desired_velocity
        #logger.debug("next_state-vel:")
        #logger.debug(next_state[:,2:4])
        next_groups = self.groups
        if groups is not None:
            next_groups = groups 
        self.update(next_state, next_groups)

    # def initial_speeds(self):
    #     return stateutils.speeds(self.ped_states[0])

    def desired_directions(self):
        return stateutils.desired_directions(self.state)[0]
    

    @staticmethod
    def capped_velocity(desired_velocity, max_velocity):
        """Scale down a desired velocity to its capped speed."""
        desired_speeds = np.linalg.norm(desired_velocity, axis=-1)
        factor = np.minimum(1.0, max_velocity / desired_speeds)
        factor[desired_speeds == 0] = 0.0
        return desired_velocity * np.expand_dims(factor, -1)

    @property
    def groups(self) -> List[List]:
        return self._groups

    @groups.setter
    def groups(self, groups: List[List]):
        if groups is None:
            self._groups = []
        else:
            self._groups = groups
        self.group_states.append(self._groups.copy())

    def has_group(self):
        return self.groups is not None

    # def get_group_by_idx(self, index: int) -> np.ndarray:
    #     return self.state[self.groups[index], :]

    def which_group(self, index: int) -> int:
        """find group index from ped index"""
        for i, group in enumerate(self.groups):
            if index in group:
                return i
        return -1


class EnvState:
    """State of the environment obstacles"""

    def __init__(self, obstacles, resolution=10):
        logger.info("Envstate init")
        self.resolution = resolution
        self.obstacles = obstacles #list

    @property # while loading: "print(EnvState.obstacles)" or "??? = EnvState.obstacles"
    def obstacles(self) -> List[np.ndarray]:  # a type
        """obstacles is a list of np.ndarray"""
        return self._obstacles #line132

    @obstacles.setter
    def obstacles(self, obstacles): # while loading: "Envstate.obstacles = ?obstacles?"
        """Input an list of (startx, endx, starty, endy) as start and end of a line"""
        if obstacles is None:
            self._obstacles = []
        else:
            self._obstacles = []
            for startx, endx, starty, endy in obstacles: #one time one obs-(a straight line)
                samples = int(np.linalg.norm((startx - endx, starty - endy)) * self.resolution) # int({sqroot[(deltaX)^2+(deltaY)^2] -> float}* resolution) -> int
                line = np.array(
                    list(
                        zip(np.linspace(startx, endx, samples), np.linspace(starty, endy, samples)) #see my picture in note  np.linspace(a,b,c) means draw c dots from a to b (both included) | zip(a,b) means mix a=[1,2,3] and b=[4,5,6] into ([1,4],[2,5],[3,6])
                    )
                ) #turn ([1,4],[2,5],[3,6]) into [[1,4],[2,5],[3,6]]
                self._obstacles.append(line) #add this obs-(a straight line)([[1,4],[2,5],[3,6]]) to Envstate.obstacles

class Pathstate:
    """Path of the dijkstra alg"""

    def __init__(self, paths):
        self.dist = []
        self.area = []
        self.coord = []
        self.wide = []
        if paths is None:
            self._path = None
        else:
            for layern, wide, path in paths:
                vectors = path[:, 0:2] - path[:, 2:4]
                cur_d = stateutils.normalize(vectors)[1]
                self.dist.append(cur_d)
                self.area.append(cur_d* wide)
                self.coord.append(path)
                self.wide.append(np.ones(path.shape[0])* wide)
            self.dist = np.concatenate(self.dist, axis = None)
            self.area = np.concatenate(self.area, axis = None)
            self.coord = np.concatenate(self.coord)
            self.wide = np.concatenate(self.wide)
            self.r_vec = stateutils.perpend(self.coord[:, 0:2] - self.coord[:, 2:4])
            self.r_vec *= np.expand_dims(self.wide, axis=1)
            self.range = np.copy(self.coord)
            self.range[:,0:2] = self.range[:,0:2] + self.r_vec
            self.range[:,2:4] = self.range[:,2:4] + self.r_vec
            self.range = np.concatenate((self.range, self.coord[:,2:4] - self.r_vec), axis=-1)
            self.range = np.concatenate((self.range, self.coord[:,0:2] - self.r_vec), axis=-1)
        #listt = []
        #for _ in goals.tolist():
        #    for __ in _:
        #        print([__[0],__[1]])
        #        c = 0
        #        for trys in listt:
        #            if trys == [__[0],__[1]]: c += 1
        #        if c == 0:
        #            listt.append([__[0],__[1]])
        #self.goal = np.array(listt)
        #self.goal = 

    @property
    def path(self):
        return self._path
    
    @path.setter
    def path(self, paths):
        self._path = paths

    #def node_desired_directions(self, state):
    #    return stateutils.node_desired_directions(state)[0]
    
    def dense(self, post, ranges):
        boool = stateutils.find(post, ranges)
        return stateutils.dense(boool, self.area)
    
    def dijkstra(self, pest, goal):
        boool = np.copy(stateutils.find(pest, self.range))
        dense = np.copy(stateutils.dense(boool, self.area))
        r_vec = np.copy(self.r_vec)
        coord = np.copy(self.coord)
        rangee = np.copy(self.range)
        area = np.copy(self.area)
        length = np.copy(self.dist)
        factor = [0.5,0.5]

        weight = dense*factor[0] + length*factor[1]
        d = {}
        c = 0
        for p1 in coord[:,0:2]:
            p2 = coord[c, 2:4]
            if d.get(str(p1)) is None:
                d[str(p1)] = {}
            if d.get(str(p2)) is None:
                d[str(p2)] = {}
            d[str(p1)].update({str(p2) : weight[c]})
            d[str(p2)].update({str(p1) : weight[c]})
            c += 1
        start_point = stateutils.render_range(boool, pest, coord)
        end_point = stateutils.render_range(stateutils.find(goal, rangee), goal, coord)
        vgoal = []
        for peo in range(pest.shape[0]):
            s = start_point[str(peo)]
            e = end_point[str(peo)]
            pos = pest[peo]
            gol = goal[peo]
            #print(pos)
            #print(gol)
            if isinstance(s,np.ndarray) and isinstance(e,np.ndarray):
                a__ = stateutils.dijkstra(d, str(s), str(e))
                if len(a__[1]) == 1 :
                    vgoal.append(str(gol))
                else :
                    vgoal.append(stateutils.dijkstra(d, str(s), str(e))[1][1])


            elif len(s) == 2 and len(e) == 1:
                tmp = []
                a0_ = stateutils.dijkstra(d, str(s[0]), str(e))
                a1_ = stateutils.dijkstra(d, str(s[1]), str(e))
                s_prefweight = stateutils.weight_single(s, pos, area, coord, length, pest, r_vec)
                s_weight = (s_prefweight[0]*factor[0] + s_prefweight[1]*factor[1], s_prefweight[2]*factor[0] + s_prefweight[3]*factor[1])
                #print("a0_",a0_)
                #print("a1_",a1_)
                tmp.append(a0_[0] + s_weight[0])
                tmp.append(a1_[0] + s_weight[1])
                tmax = min(tmp)
                ans = stateutils.dijkstra(d, str(s[tmp.index(tmax)]), str(e))
                if len(ans[1]) == 1 :
                    vgoal.append(str(gol))
                else :
                    vgoal.append(ans[1][0])


            elif isinstance(s,np.ndarray) and isinstance(e,list):
                tmp = []
                a_0 = stateutils.dijkstra(d, str(s), str(e[0]))
                a_1 = stateutils.dijkstra(d, str(s), str(e[1]))
                e_prefweight = stateutils.weight_single(e, pos, area, coord, length, pest, r_vec)
                e_weight = (e_prefweight[0]*factor[0] + e_prefweight[1]*factor[1], e_prefweight[2]*factor[0] + e_prefweight[3]*factor[1])
                #print("a_0",a_0)
                #print("a_1",a_1)
                tmp.append(a_0[0] + e_weight[0])
                tmp.append(a_1[0] + e_weight[1])
                tmax = min(tmp)
                ans = stateutils.dijkstra(d, str(s), str(e[tmp.index(tmax)]))
                if len(ans[1]) == 1 :
                    vgoal.append(str(gol))
                else :
                    vgoal.append(ans[1][1])


            elif isinstance(s,list) and isinstance(e,list):
                tmp = []
                #print(type(s))
                #print(type(e))
                #print(d[str(s[0])])
                #print(d[str(s[1])])
                #print(d[str(e[0])])
                #print(d[str(e[1])])
                s_prefweight = stateutils.weight_single(s, pos, self.area, coord, length, pest, r_vec)
                e_prefweight = stateutils.weight_single(e, pos, self.area, coord, length, pest, r_vec)
                s_weight = (s_prefweight[0]*factor[0] + s_prefweight[1]*factor[1], s_prefweight[2]*factor[0] + s_prefweight[3]*factor[1])
                e_weight = (e_prefweight[0]*factor[0] + e_prefweight[1]*factor[1], e_prefweight[2]*factor[0] + e_prefweight[3]*factor[1])
                a00 = stateutils.dijkstra(d, str(s[0]), str(e[0]))
                a10 = stateutils.dijkstra(d, str(s[1]), str(e[0]))
                a01 = stateutils.dijkstra(d, str(s[0]), str(e[1]))
                a11 = stateutils.dijkstra(d, str(s[1]), str(e[1]))
                #print("a00",a00)
                #print("a10",a10)
                #print("a01",a01)
                #print("a11",a11)
                tmp.append(a00[0] + s_weight[0] + e_weight[0])
                tmp.append(a10[0] + s_weight[1] + e_weight[0])
                tmp.append(a01[0] + s_weight[0] + e_weight[1])
                tmp.append(a11[0] + s_weight[1] + e_weight[1])
                tmax = min(tmp)
                idx1 = 0
                idx2 = 0
                if tmp.index(tmax) == 1 : idx1 = 1
                elif tmp.index(tmax) == 2 : idx2 = 1
                elif tmp.index(tmax) == 3 : idx2, idx1 = 1, 1
                ans = stateutils.dijkstra(d, str(s[idx1]), str(e[idx2]))
                if len(ans[1]) == 1 :
                    vgoal.append(str(gol))
                else :
                    vgoal.append(ans[1][0])
        #print(vgoal)
        a = []
        for _ in vgoal:
            #print(_.split(' ')[-1])#.split(']')[0])
            #print([float(_.split(' ')[0].split('[')[1]),float(_.split(' ')[-1].split(']')[0])])
            a.append([float(_.split(' ')[0].split('[')[1]),float(_.split(' ')[-1].split(']')[0])])
        #print(np.array(a))
        
        return np.array(a)

