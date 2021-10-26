import numpy as np
import quaternion

def vec_to_trans(vec):
    a = np.identity(len(vec)+1)
    a[:len(vec),len(vec)] = vec
    return a

def get_square_horizon(base_xy, radius, z_angle=0):
    d1 = [1,1,-1,-1]
    d2 = [1,-1,-1,1]
    poss = np.array([np.quaternion(1,d1[i]*radius,d2[i]*radius,0) for i in range(4)])
    q = quaternion.from_euler_angles(0,0,z_angle)
    pq = q*poss*q.conj()
    return np.array([[p.x, p.y] for p in pq])+base_xy

def add_dimension(vecs, value=0):
    return np.concatenate([vecs[np.newaxis,:,i].T if i<len(vecs[0]) else np.full([len(vecs),1], value) for i in range(len(vecs[0])+1)], axis=1)

