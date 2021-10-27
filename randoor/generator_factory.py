import abc
import numpy as np
from numpy.lib.arraysetops import isin
import shapely
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from multiprocessing import Pool, Array
from contextlib import closing

from .utils import vec_to_transform_matrix, radian_to_rotation_matrix
from spawner.poly import get_moved_poly

class RoomGeneratorFactory(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def generate_new(self):
        pass

class RoomConfig(object):
    def __init__(self):
        # config[tag] = {collision: bool, baseshape: Polygon, positions: [(x,y,yaw)]}
        self.config = dict()
        # polygons[tag] = [Polygon]
        self.polygons = dict()

        self.conf_tag_collision = 'collision'
        self.conf_tag_positions = 'positions'
        self.conf_tag_baseshape = 'baseshape'
    
    @abc.abstractmethod
    def prepare(self):
        pass

    def register(self, tag, base_shape, count):
        self.config[tag] = {
            self.conf_tag_collision: False,
            self.conf_tag_positions: np.zeros([count,3]),
            self.conf_tag_baseshape: base_shape,
        }
        self.polygons[tag] = np.empty([count], dtype=object)
    
    def set_collision(self, tag, is_collision):
        self.config[tag][self.conf_tag_collision] = is_collision
    
    def set_positions(self, tag, x_y_yaw):
        self.config[tag][self.conf_tag_positions] = tuple(x_y_yaw)

    def set_baseshape(self, tag, base_shape):
        self.config[tag][self.conf_tag_baseshape] = base_shape

    def set_config_all(self, tag, is_collision, x_y_yaw, base_shape):
        self.set_collision(tag, is_collision)
        self.set_positions(tag, x_y_yaw)
        self.set_baseshape(tag, base_shape)

    def set_config(self, tag, is_collision, x_y_yaw):
        self.set_collision(tag, is_collision)
        self.set_positions(tag, x_y_yaw)

    def set_polygons_direct(self, tag, polygons):
        self.polygons[tag] = polygons

    def set_polygons_auto(self, tag):
        assert tag in self.config.keys(), 'not registered tag'
        base_shape = self.config[tag][self.conf_tag_baseshape]
        x_y_yaw = self.config[tag][self.conf_tag_positions]
        polys = [get_moved_poly(base_shape, p[0], p[1], p[2]) for p in x_y_yaw]
        self.set_polygons_direct(tag, polys)

    def gather_polygon_from_config(self, conf_filter=None):
        l = []
        for tag, conf in self.config.items():
            flag = (conf_filter is None) or (conf_filter(tag, conf))
            if flag:
                l.extend(self.polygons[tag])
        return l

    def get_space_poly(self, exterior_tag, poly_index=0):
        exte_pol = self.polygons[exterior_tag][poly_index]
        assert len(exte_pol.interiors) == 1, 'exterior_tag should be perforated shape'
        inte_pol = Polygon(exte_pol.interiors[0])
        
        conf_filter = lambda tag, conf: (conf[self.conf_tag_collision]) and (tag != exterior_tag)
        polys = unary_union(self.gather_polygon_from_config(conf_filter))
        if isinstance(polys, Polygon):
            holes = polys.exterior.coords
        else:
            holes = [p.exterior.coords for p in polys]

        return Polygon(inte_pol.exterior.coords, holes)

    def get_collision_poly(self):
        l = []
        for tag, conf in self.config.items():
            if conf[self.conf_tag_collision]:
                l.extend(self.polygons[tag])
        return unary_union(l)

    @abc.abstractmethod
    def get_freespace_poly(self):
        pass

    @abc.abstractmethod
    def get_occupancy_grid(self, freespace_poly, origin_pos=(0,0), origin_ori=0, resolution=0.050, map_size=512, pass_color=255, obs_color=0):
        if origin_pos == (0,0) and origin_ori == 0:
            corrected = freespace_poly
        else:
            mat1 = vec_to_transform_matrix(origin_pos)
            mat2 = radian_to_rotation_matrix(origin_ori)
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