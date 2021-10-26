import abc
import numpy as np
from numpy.lib.arraysetops import isin
import shapely
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from multiprocessing import Pool, Array
from contextlib import closing

from .utils import vec_to_trans

class RoomGeneratorFactory(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_new(self):
        pass

class RoomConfig(object):
    def __init__(self):
        # config[tag] = {collision: bool, positions: [(x,y,yaw)], polygons: [Polygon]}
        self.config = dict()

        self.conf_tag_collision = 'collision'
        self.conf_tag_positions = 'positions'
        self.conf_tag_polygons = 'polygons'
    
    @abc.abstractmethod
    def prepare(self):
        pass

    def register_config(self, tag, count):
        self.config[tag] = {
            self.conf_tag_collision: False,
            self.conf_tag_positions: np.zeros([count,3]),
            self.conf_tag_polygons: np.empty([count], dtype=object),
        }
    
    def set_collision(self, tag, collision):
        self.config[tag][self.conf_tag_collision] = collision
    
    def set_positions(self, tag, positions):
        self.config[tag][self.conf_tag_positions] = positions

    def set_polygons(self, tag, polygons):
        self.config[tag][self.conf_tag_polygons] = polygons

    def set_config(self, tag, collision, positions, polygons):
        self.config[tag][self.conf_tag_collision] = collision
        self.config[tag][self.conf_tag_positions] = positions
        self.config[tag][self.conf_tag_polygons] = polygons

    def gather_polygon_from_config(self, conf_filter=None):
        l = []
        for tag, conf in self.config.items():
            flag = (conf_filter is None) or (conf_filter(tag, conf))
            if flag:
                l.extend(conf[self.polygon_group])
        return l

    def get_space_poly(self, exterior_tag, poly_index=0, exclude_tags=None):
        exterior = self.config[exterior_tag][self.conf_tag_polygons][poly_index]
        assert len(exterior.interiors) == 1, 'exterior_tag should be perforated shape'
        
        conf_filter = lambda tag, conf: (conf[self.conf_tag_collision]) and (tag != exterior_tag)
        polys = unary_union(self.gather_polygon_from_config(conf_filter))
        if isinstance(polys, Polygon):
            holes = [polys.exterior.coords]
        else:
            holes = [p.exterior.coords for p in polys]

        return Polygon(exterior.exterior.coords, holes)

    def get_collision_poly(self):
        l = []
        for conf in self.config.values():
            if conf[self.conf_tag_collision]:
                l.extend(conf[self.conf_tag_polygons])
        return unary_union(l)

    @abc.abstractmethod
    def get_freespace_poly(self):
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
        
        data = np.full([map_size*map_size], obs_color, dtype=np.uint8)

        xl = Array('d', xc)
        yl = Array('d', yc)
        
        global mu

        def mu(i):
            return corrected.contains(Point(xl[i], yl[i]))

        with closing(Pool()) as pool:
            data[pool.map(mu, range(len(xc)))] = pass_color
    
        return data