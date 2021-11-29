import numpy as np
import quaternion

def vec_to_transform_matrix(vec):
    a = np.identity(len(vec)+1)
    a[:len(vec),len(vec)] = vec
    return a

def radian_to_rotation_matrix(radian):
    return np.array([
        [np.cos(radian), -np.sin(radian), 0],
        [np.sin(radian), np.cos(radian), 0],
        [0,0,1]
    ])

def get_square_horizon(base_xy, radius, z_angle=0):
    d1 = [1,1,-1,-1]
    d2 = [1,-1,-1,1]
    poss = np.array([np.quaternion(1,d1[i]*radius,d2[i]*radius,0) for i in range(4)])
    q = quaternion.from_euler_angles(0,0,z_angle)
    pq = q*poss*q.conj()
    return np.array([[p.x, p.y] for p in pq])+base_xy

def add_dimension(vecs, value=0):
    return np.concatenate([vecs[np.newaxis,:,i].T if i<len(vecs[0]) else np.full([len(vecs),1], value) for i in range(len(vecs[0])+1)], axis=1)

def convert_180(rad360):
    '''
    [0,360] -> [-180,180]
    '''
    return ((rad360 - np.pi) % (np.pi*2)) - np.pi

def convert_360(rad180):
    '''
    [-180,180] -> [0,360]
    '''
    return rad180 % (np.pi*2)

def transform_2d(target_x, target_y, target_yaw, origin_x, origin_y, origin_yaw):
    t_mat = np.identity(3)
    t_mat[:2,2] = [-origin_x, -origin_y]
    r_mat = np.array([
                [np.cos(-origin_yaw), -np.sin(-origin_yaw), 0],
                [np.sin(-origin_yaw), np.cos(-origin_yaw), 0],
                [0, 0, 1]
            ])
    af = np.dot(r_mat, t_mat)
    target_xy = np.array([target_x, target_y, 1])
    transed_xy = np.dot(af, target_xy)
    
    return transed_xy[0], transed_xy[1], convert_180(target_yaw - origin_yaw)

def get_affine_tf_rt(x, y, yaw):
    """
    Notice: Calculation with transform -> rotate
    """
    rotate = radian_to_rotation_matrix(yaw)
    # trans = vec_to_transform_matrix([y,-x])
    trans = vec_to_transform_matrix([x,y])
    affine = np.dot(trans, rotate)
    return affine

def get_affine_rt_tf(x, y, yaw):
    """
    Notice: rotate -> transform
    """
    rotate = radian_to_rotation_matrix(yaw)
    # trans = vec_to_transform_matrix([y,-x])
    trans = vec_to_transform_matrix([x,y])
    affine = np.dot(rotate, trans)
    return affine