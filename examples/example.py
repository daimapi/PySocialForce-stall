#utf-8
from pathlib import Path
import numpy as np
import pysocialforce as psf


if __name__ == "__main__":
    # initial states, each entry is the position, velocity and goal of a pedestrian in the form of 
    #[px, py, vx, vy, gx, gy, gt, gx2, gy2]
    initial_state = np.array(
        [
            [0.0, 10, -0.5, -0.5, 0.0, 0.0, 3, 0.0, 10],
            [0.5, 10, -0.5, -0.5, 0.5, 0.0, 3, 0.5, 10],
            [0.0, 0.0, 0.0, 0.5, 1.0, 10.0, 3, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.5, 2.0, 10.0, 3, 1.0, 0.0],
            [2.0, 0.0, 0.0, 0.5, 3.0, 10.0, 3, 2.0, 0.0],
            [3.0, 0.0, 0.0, 0.5, 4.0, 10.0, 3, 3.0, 0.0],
        ]
    )
    #initial_state = np.array(
    #    [
    #        [0.0, 10, -0.5, -0.5, 0.0, 0.0, 1.0, 10.0],
    #        [0.5, 10, -0.5, -0.5, 0.5, 0.0, 0.0, 10.0],
    #    ]
    #)
    # social groups informoation is represented as lists of indices of the state array)
    # list of linear obstacles given in the form of (x_min, x_max, y_min, y_max)
    groups = [[1],[2],[3],[4],[5],[0]] #分組機制(依 initial_state 之 array 行數
    #groups = [[0]]
    # obs = [[-1, -1, -1, 11], [3, 3, -1, 11]]
    obs = [[1, 2, 7, 8]]
    # obs = None
    # initiate the simulator,
    s = psf.Simulator(
        initial_state,
        obstacles = obs,
        groups = groups,
                config_file = Path(__file__).resolve().parent.joinpath("example.toml"),
    ) #呼叫 simulator.py 的 __init__
    
    # update n steps
    n = 80
    s.step(n) #no difference
    name = "images/exmaple__agents=" + str(initial_state.shape[0]) + "_n=" + str(n) + ""
    with psf.plot.SceneVisualizer(s, name) as sv:
        sv.animate()
        # sv.plot()
