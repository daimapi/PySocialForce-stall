from xml.dom import minidom
from svg.path import parse_path #Pylint sucks
import numpy as np
from pysocialforce.utils import stateutils,logger
from shapely.geometry import Point, Polygon
############################################################################################reader in example
laywide = {
    "1":"3",
    "2":"5",
    "3":"10",
}
init_path = []
doc = minidom.parse('shortest paths.svg')
for i, svg in enumerate(doc.getElementsByTagName('svg')):
    h = int(svg.getAttribute('height').split('mm')[0])
for ilayer, layer in enumerate(doc.getElementsByTagName('g')):
    npobds = np.array([[]])
    layern = layer.getAttribute('id').split('layer')[1]
    for ipath, path in enumerate(layer.getElementsByTagName('path')):
        d = path.getAttribute('d')
        parsed = parse_path(d)
        for obj in parsed:
            if type(obj).__name__ != 'Move':
                #reg = np.expand_dims(np.array([round(obj.start.real, 3),
                #                               h - round(obj.start.imag, 3) ,
                #                               round(obj.end.real, 3) , 
                #                               h - round(obj.end.imag, 3)]), axis=0)
                reg = np.expand_dims(np.array([obj.start.real,
                                               h - obj.start.imag,
                                               obj.end.real , 
                                               h - obj.end.imag]), axis=0)
                if npobds.shape == (1,0):
                    npobds = reg
                else:
                    npobds = np.insert(npobds, 0, reg, axis=0)
    paths = []
    #c = 0
    for _ in list(npobds):
        item = []
        #c += 1
        for i in _:
            item.append(i)
        paths.append(item)
    init_path.append([int(layern), int(laywide[layern]), np.array(paths)])
#print(init_path)
    # paths_array = [[x1, y1, x2, y2],
    #                [x1, y1, x2, y2],...]
    # init_path = [[layern, wide, coord],
    #              [layern, wide, coord],...]
############################################################################################
#######################################################################################scene.Path init
dist = []
area = []
coord = []
wide = []
for layern, wides, path in init_path:
    print(layern)
    print(wides)
    print(path)
    vectors = path[:, 0:2] - path[:, 2:4]
    cur_dist = stateutils.normalize(vectors)[1]
    dist.append(cur_dist)
    area.append(cur_dist* wides)
    coord.append(path)
    wide.append(np.ones(path.shape[0])* wides)
dist = np.concatenate(dist, axis = None)
area = np.concatenate(area, axis = None)
coord = np.concatenate(coord)
wide = np.concatenate(wide)
r_vec = stateutils.perpend(coord[:, 0:2] - coord[:, 2:4])
#print("aaaaaaaaaaaaaaaaaaaa",r_vec)
r_vec *= np.expand_dims(wide, axis=1)
rangee = np.copy(coord)
rangee[:,0:2] = rangee[:,0:2] + r_vec
rangee[:,2:4] = rangee[:,2:4] + r_vec
rangee = np.concatenate((rangee, coord[:,2:4] - r_vec), axis=-1)
rangee = np.concatenate((rangee, coord[:,0:2] - r_vec), axis=-1)
#######################################################################################
#print(dist)
#print(area)
#print(coord)
#print(rangee)

#print(layern)
##print(c)
#print(paths)
#print(init_path)

#for ipath, path in enumerate(doc.getElementsByTagName('path')):
#    d = path.getAttribute('d')
#    parsed = parse_path(d)
#    for obj in parsed:
#        if type(obj).__name__ != 'Move':
#            reg = np.expand_dims(np.array([round(obj.start.real, 3),round(obj.end.real, 3) ,h - round(obj.start.imag, 3) , h - round(obj.end.imag, 3)]), axis=0)
#            if npobds.shape == (1,0):
#                npobds = reg
#            else:
#                npobds = np.insert(npobds, 0, reg, axis=0)
doc.unlink()

#print("amount = ", c)
graph = {}
print()

graph={
  's':{'a':8,'b':4},
  'a':{'b':4},
  'b':{'a':3,'c':2,'d':5},
  'c':{'d':2},
  'd':{}
}
for k1, v1 in graph.items():
    for k2, v2 in v1.items():
            print(k1,k2,v2)
pos = np.array([[ 13,  61,  0,   0,   29,   30,   0,   0, ],
                [ 37,  35,  0,   0,   29,   30,   0,   0, ]
                ],dtype=float)
#print(boool)
#print(stateutils.dense(boool, area))

pest = np.copy(pos[:,0:2])
goal = np.copy(pos[:,4:6])
length = np.copy(dist)
boool = stateutils.find(pest, rangee)
dense = stateutils.dense(boool, area)
factor = [0.5,0.5]
print("boool",boool)
print("pest",pest)
print("coord",coord)
print("dense",dense)
print("length",length)
print("rangee",rangee)
print("area",area)
print("r_vec",r_vec)
print("factors_d_l",factor)

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

#print(d)
#logger.info(d)


#pest boool coord############################################################################################
#start_point = {}
#for peo in range(pest.shape[0]):
#    a = boool[:, peo]
#    #print(rangee[a])
#    print(coord[a])
#    #print(r_vec[a])
#    #print(stateutils.perpend(coord[:, 0:2] - coord[:, 2:4])[a])
#    #point = coord(n,2) ; 
#    if coord[a].size > 4:
#        copy = np.copy(coord[a])
#        multi_pos = copy.reshape((copy.shape[0], 2, 2))
#        x0y0 = multi_pos[0, 0]
#        booll = np.repeat(np.array([x0y0,x0y0]), 2, axis=0).reshape(copy.shape[0], 2, 2) == multi_pos
#        b = 0
#        for _ in booll:
#            c = 0
#            for __ in _:
#                if __.sum() == 2:
#                    c += 1
#            b += c
#        if b == copy.shape[0]:#-->first
#            start_point[str(peo)] = x0y0
#        else:#-->second
#            start_point[str(peo)] = x0y0
#    else:
#        copy = coord[a]
#        start_point[str(peo)] = [copy[0,0:2],copy[0,2:4]]
#print(start_point)
start_point = stateutils.render_range(boool, pest, coord)
print(start_point)
#pest boool coord############################################################################################
##############################################################scene.Path init (abandoned)
#s = np.array(
#    [
#        [[29, 30, 0 ], [49, 30, 2] , [95, 73, 3] ],
#        #[[0.5, 0.0, 3] , [0.5, 10, 3] , [0.5, 0.0, 3] ],
#        #[[1.0, 10.0, 3], [0.0, 0.0, 1], [1.0, 10.0, 3]],
#        #[[2.0, 10.0, 3], [1.0, 0.0, 1], [2.0, 10.0, 3]],
#        #[[3.0, 10.0, 3], [2.0, 0.0, 1], [3.0, 10.0, 3]],
#        #[[4.0, 10.0, 3], [3.0, 0.0, 1], [4.0, 10.0, 3]],
#    ]
#)
#listt = []
#for _ in s.tolist():
#    for __ in _:
#        c = 0
#        for trys in listt:
#            if trys == [__[0],__[1]]: c += 1
#        if c == 0:
#            listt.append([__[0],__[1]])
#goal = np.array(listt)
#print(goal)
##############################################################
end_point = stateutils.render_range(stateutils.find(goal, rangee), goal, coord)
print(end_point)



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
            

    elif isinstance(s,list) and isinstance(e,np.ndarray):
        tmp = []
        a0_ = stateutils.dijkstra(d, str(s[0]), str(e))
        a1_ = stateutils.dijkstra(d, str(s[1]), str(e))
        s_prefweight = stateutils.weight_single(s, pos, area, coord, length, pest, r_vec)
        s_weight = (s_prefweight[0]*factor[0] + s_prefweight[1]*factor[1], s_prefweight[2]*factor[0] + s_prefweight[3]*factor[1])
        print("a0_",a0_)
        print("a1_",a1_)
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
        print("a_0",a_0)
        print("a_1",a_1)
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
        s_prefweight = stateutils.weight_single(s, pos, area, coord, length, pest, r_vec)
        e_prefweight = stateutils.weight_single(e, pos, area, coord, length, pest, r_vec)
        s_weight = (s_prefweight[0]*factor[0] + s_prefweight[1]*factor[1], s_prefweight[2]*factor[0] + s_prefweight[3]*factor[1])
        e_weight = (e_prefweight[0]*factor[0] + e_prefweight[1]*factor[1], e_prefweight[2]*factor[0] + e_prefweight[3]*factor[1])
        a00 = stateutils.dijkstra(d, str(s[0]), str(e[0]))
        a10 = stateutils.dijkstra(d, str(s[1]), str(e[0]))
        a01 = stateutils.dijkstra(d, str(s[0]), str(e[1]))
        a11 = stateutils.dijkstra(d, str(s[1]), str(e[1]))
        print("a00",a00)
        print("a10",a10)
        print("a01",a01)
        print("a11",a11)
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
print(vgoal)
a = []
for _ in vgoal:
    #print(_.split(' ')[-1])#.split(']')[0])
    #print([float(_.split(' ')[0].split('[')[1]),float(_.split(' ')[-1].split(']')[0])])
    a.append([float(_.split(' ')[0].split('[')[1]),float(_.split(' ')[-1].split(']')[0])])
print(np.array(a))
    