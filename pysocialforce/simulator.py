"""Simulator"""

# coding=utf-8

"""Synthetic pedestrian behavior with social groups simulation according to the Extended Social Force model.

See Helbing and Molnár 1998 and Moussaïd et al. 2010
"""
import numpy as np
import csv
from pysocialforce.utils import DefaultConfig, logger
from pysocialforce.scene import PedState, EnvState
from pysocialforce import forces

class Simulator:
    """Simulate social force model.

    ...

    Attributes
    ----------
    state : np.ndarray [n, 6] or [n, 7]
       Each entry represents a pedestrian state, (x, y, v_x, v_y, d_x, d_y, d2_x, d2_y, [tau])
    obstacles : np.ndarray
        Environmental obstacles
    groups : List of Lists
        Group members are denoted by their indices in the state
    config : Dict
        Loaded from a toml config file
    max_speeds : np.ndarray
        Maximum speed of pedestrians
    forces : List
        Forces to factor in during navigation

    Methods
    ---------
    capped_velocity(desired_velcity)
        Scale down a desired velocity to its capped speed
    step()
        Make one step
    """

    def __init__(self, state, goals, groups=None, obstacles=None, config_file=None):
        
        logger.info("simulator init")
        self.config = DefaultConfig()
        if config_file:
            self.config.load_config(config_file) # load config_file 
        
        # TODO: load obstacles from config
        self.scene_config = self.config.sub_config("scene")
        # initiate obstacles
        self.env = EnvState(obstacles, self.config("resolution", 10.0))   # env = environment 11/11

        # initiate agents
        self.peds = PedState(state, goals, groups, self.config) 

        # construct forces
        self.forces = self.make_forces(self.config)
        logger.info("simulator end init")

    def make_forces(self, force_configs):
        """Construct forces"""
        logger.info("making force (init)")
        force_list = [
            forces.DesiredForce(),
            forces.SocialForce(),
            forces.ObstacleForce(),
            # forces.PedRepulsiveForce(),
            # forces.SpaceRepulsiveForce(),
        ]
        group_forces = [
            forces.GroupCoherenceForceAlt(),
            forces.GroupRepulsiveForce(),
            forces.GroupGazeForceAlt(),
        ]
        if self.scene_config("enable_group"):
            force_list += group_forces

        # initiate forces
        for force in force_list:
            force.init(self, force_configs)  # more details in config.py
        
        logger.info("done force init:")
        #logger.info(force_list)

        return force_list

    def compute_forces(self):
        """compute forces"""
        logger.info("start compute_forces --> get_force and sum up all force")
        m = np.squeeze(self.peds.num(), axis=-1) > 0
        logger.debug("m: ")
        logger.debug(m)
        force = map(lambda x: x.get_force(), self.forces)
        if (m.sum() == 0):
            return np.resize([0],(self.peds.size(),2))
        force = sum(force)
        
        ans = np.zeros((m.shape[0],2))
        for a in range(m.shape[0]):
            if m[a]:
                ans[a] = force[0]
                force = np.delete(force, 0, 0)
        logger.debug("force:")
        logger.debug(ans)
        #m = np.squeeze(self.peds.num(), axis=-1) > 0##############調回全長
        #
        #ans = np.zeros((m.shape[0],force.shape[0]))
        #force = np.rot90(force,3)
        #for a in range(m.shape[0]):
        #    if m[a]:
        #        ans[a] = force[0]
        #        force = np.delete(force, 0, 0)
        return ans#np.rot90(ans)

    def get_states(self):
        """Expose whole state"""
        return self.peds.get_states()

    def get_length(self):
        """Get simulation length"""
        return len(self.get_states()[0])

    def get_obstacles(self):
        return self.env.obstacles

    def step_once(self):
        """step once"""
        logger.debug("step once")
        self.peds.step(self.compute_forces())

    def step(self, n=1):
        """Step n time"""
        for _ in range(n):
            logger.info(f"steping:{_}")
            self.step_once()
        logger.info('finish caculation')

        with open('state.csv', 'w+', newline='') as csvfile:
            filewriter = csv.writer(csvfile)
            for row in self.peds.get_pure_state():
                for peo in row:
                    filewriter.writerow(peo)
                    #for eachsta in peo:
                        #filewriter.writerow(eachsta)

        return self
