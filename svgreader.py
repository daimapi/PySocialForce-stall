#=============================================================
#**coding: UTF-8**
#by 
#
#
#2024/2/1 下午5:24:54
#=============================================================
from xml.dom import minidom
from svg.path import parse_path
import numpy as np

obs = np.array([[]])
doc = minidom.parse('isuck.svg')
for i, svg in enumerate(doc.getElementsByTagName('svg')):
    h = int(svg.getAttribute('height').split('mm')[0])
for ipath, path in enumerate(doc.getElementsByTagName('path')):
    #print('Path', ipath)
    d = path.getAttribute('d')
    parsed = parse_path(d)
    #print('Objects:\n', parsed, '\n' + '-' * 20)
    for obj in parsed:
        if (type(obj).__name__ != 'Move'):
            reg = np.expand_dims(np.array([round(obj.start.real, 3),round(obj.end.real, 3) ,h - round(obj.start.imag, 3) , h - round(obj.end.imag, 3)]), axis=0)
            if (obs.shape == (1,0)):
                obs = reg
            else:
                obs = np.insert(obs, 0, reg, axis=0)
doc.unlink()
#print(obs)
#print(obs.shape)
ans = []
c=0
for _ in list(obs):
    item = []
    for i in _:
        item.append(i)
    ans.append(item)
    c = c + 1
print(c)
    
init_state = np.array(
    [
        [0.0, 10, -0.5, -0.5, 0],
        [0.5, 10, -0.5, -0.5, 1],
        [0.0, 0.0, 0.0, 0.5, 5],
        #[1.0, 0.0, 0.0, 0.5, 10],
        #[2.0, 0.0, 0.0, 0.5, 13],
        #[3.0, 0.0, 0.0, 0.5, 17],
    ]
)
n = [[0]]
for _ in range(init_state.shape[0]-1):
    _ = _ + 1
    n.append([_])
print(n)