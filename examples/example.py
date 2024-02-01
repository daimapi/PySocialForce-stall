#utf-8
from pathlib import Path
from xml.dom import minidom
from svg.path import parse_path #Pylint sucks
import numpy as np
import pysocialforce as psf

if __name__ == "__main__":
    npobds = np.array([[]])
    doc = minidom.parse('isuck.svg')
    for i, svg in enumerate(doc.getElementsByTagName('svg')):
        h = int(svg.getAttribute('height').split('mm')[0])
    for ipath, path in enumerate(doc.getElementsByTagName('path')):
        d = path.getAttribute('d')
        parsed = parse_path(d)
        for obj in parsed:
            if (type(obj).__name__ != 'Move'):
                reg = np.expand_dims(np.array([round(obj.start.real, 3),round(obj.end.real, 3) ,h - round(obj.start.imag, 3) , h - round(obj.end.imag, 3)]), axis=0)
                if (npobds.shape == (1,0)):
                    npobds = reg
                else:
                    npobds = np.insert(npobds, 0, reg, axis=0)
    doc.unlink()
    obs = []
    for _ in list(npobds):
        item = []
        for i in _:
            item.append(i)
        obs.append(item)
    # initial states, each entry is the position, velocity and goal of a pedestrian in the form of 
    #[px, py, vx, vy, pt],
    init_state = np.array(
        [
            [0.0, 10, -0.5, -0.5, 0],
            [0.5, 10, -0.5, -0.5, 1],
            [0.0, 0.0, 0.0, 0.5, 5],
            [1.0, 0.0, 0.0, 0.5, 10],
            [2.0, 0.0, 0.0, 0.5, 13],
            [3.0, 0.0, 0.0, 0.5, 17],
        ]
    )
    #initial_state = np.array(
    #    [
    #        [0.0, 10, -0.5, -0.5, 0.0, 0.0, 1.0, 10.0],
    #        [0.5, 10, -0.5, -0.5, 0.5, 0.0, 0.0, 10.0],
    #    ]
    #)
    #[[g1x,g1y,g1t],[g2x,g2y,g2t]...],
    init_goals = np.array(
        [
            [[0.0, 0.0, 0] , [0.0, 10, 2] , [0.0, 0.0, 3] ],
            [[0.5, 0.0, 3] , [0.5, 10, 3] , [0.5, 0.0, 3] ],
            [[1.0, 10.0, 3], [0.0, 0.0, 1], [1.0, 10.0, 3]],
            [[2.0, 10.0, 3], [1.0, 0.0, 1], [2.0, 10.0, 3]],
            [[3.0, 10.0, 3], [2.0, 0.0, 1], [3.0, 10.0, 3]],
            [[4.0, 10.0, 3], [3.0, 0.0, 1], [4.0, 10.0, 3]],
        ]
    )
    npobds = np.array([[]])
    doc = minidom.parse('isuck.svg')
    for i, svg in enumerate(doc.getElementsByTagName('svg')):
        h = int(svg.getAttribute('height').split('mm')[0])
    for ipath, path in enumerate(doc.getElementsByTagName('path')):
        d = path.getAttribute('d')
        parsed = parse_path(d)
        for obj in parsed:
            if type(obj).__name__ != 'Move':
                reg = np.expand_dims(np.array([round(obj.start.real, 3),round(obj.end.real, 3) ,h - round(obj.start.imag, 3) , h - round(obj.end.imag, 3)]), axis=0)
                if npobds.shape == (1,0):
                    npobds = reg
                else:
                    npobds = np.insert(npobds, 0, reg, axis=0)
    doc.unlink()
    obs = []
    for _ in list(npobds):
        item = []
        for i in _:
            item.append(i)
        obs.append(item)
    #obs = [
    #        [  1,   2,   7,   8],
    #        [ 20,  20,  20, -20],
    #        [ 20, -20, -20, -20],
    #        [-20, -20, -20,  20],
    #        [-20,  20,  20,  20],
    #      ]
    #print(obs)
    # list of linear obstacles given in the form of (x1, x2, y1, y2)
    #obs = [[-1, -1, -1, 11], [3, 3, -1, 11]]
    # obs = None
    
    
    # social groups informoation is represented as lists of indices of the state array)
    groups = [[1],[2],[3],[4],[5],[0]] #分組機制(依 initial_state 之 array 行數
    #groups = [[0]]
    
    # initiate the simulator,
    s = psf.Simulator(
        init_state,
        init_goals,
        obstacles = obs,
        groups = groups,
                config_file = Path(__file__).resolve().parent.joinpath("example.toml"),
    ) #呼叫 simulator.py 的 __init__
    
    # update n steps
    n = 200
    s.step(n) #no difference
    name = "images/exmaple__agents=" + str(init_state.shape[0]) + "_n=" + str(n) + ""
    with psf.plot.SceneVisualizer(s, name) as sv:
        sv.animate()
        # sv.plot()
