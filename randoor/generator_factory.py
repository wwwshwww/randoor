import abc
import quaternion
import numpy as np
import shapely
from shapely.geometry import Polygon, Point
from multiprocessing import Pool, Array
from contextlib import closing

from .geometric_utils import vec_to_trans

class RoomGeneratorFactory(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_new(self):
        pass

class RoomConfig(object):
    def __init__(self):
        self.spawn_config = dict() # 'tag': {'positions': ~, 'orientations': ~}
        self.point_group = dict()
        self.polygon_group = dict()
        self.config_tags = list()
    
    @abc.abstractmethod
    def prepare_model_manager(self):
        pass

    def initialize_config(self, tag, count):
        self.config_tags.append(tag)
        self.spawn_config[tag] = dict(
            positions=np.zeros([count,3]),
            orientations=np.zeros([count,3])
        )

    def register_positions(self, tag, positions):
        self.spawn_config[tag]['positions'] = positions

    def register_orientations(self, tag, orientations):
        self.spawn_config[tag]['orientations'] = orientations

    @abc.abstractmethod
    def get_freespace_poly(self, exterior_wall_tag, exclude_tags=[None]):
        pass

    @abc.abstractmethod
    def get_occupancy_grid(self, freespace_poly, origin_pos=(0,0), origin_ori=0, resolution=0.050, map_size=512, pass_color=255, obs_color=0):
        if origin_pos == (0,0) and origin_ori == 0:
            corrected = freespace_poly
        else:
            mat1 = vec_to_trans(origin_pos)
            mat2 = np.array([
                [np.cos(origin_ori), -np.sin(origin_ori), 0],
                [np.sin(origin_ori), np.cos(origin_ori), 0],
                [0, 0, 1]
            ])
            af = np.dot(mat2,mat1)
            print(af)
            corrected = shapely.affinity.affine_transform(freespace_poly, [af[0,0],af[0,1],af[1,0],af[1,1],af[0,2],af[1,2]])
    
        half_length = (map_size * resolution) / 2
        lin = np.linspace(-half_length, half_length, map_size)
        xx, yy = np.meshgrid(lin, lin)
        xc = xx.flatten()
        yc = yy.flatten()
        
        data = np.full([map_size*map_size], 0, dtype=np.uint8)

        xl = Array('d', xc)
        yl = Array('d', yc)
        
        global mu

        def mu(i):
            return corrected.contains(Point(xl[i], yl[i]))

        with closing(Pool()) as pool:
            data[pool.map(mu, range(len(xc)))] = 255
    
        return data