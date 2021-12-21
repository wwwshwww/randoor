import abc
import numpy as np
from numpy.lib.arraysetops import isin
import shapely
from shapely.geometry import Polygon, Point, MultiPolygon
from shapely.ops import unary_union
from multiprocessing import Pool, Array
from contextlib import closing

from .utils import vec_to_transform_matrix, radian_to_rotation_matrix
from .spawner.poly import get_moved_poly_rt_tf, get_moved_poly_tf_rt, get_affine_rt_tf
from randoor.spawner import poly

class RoomGeneratorFactory(object):
    def __init__(self):
        pass

    @staticmethod
    def merge_config(base_instance, target_instance):
        # target_instance.config.update(base_instance.config)
        # target_instance.polygons.update(target_instance.polygons)
        for tag, config in base_instance.config.items():
            target_instance.config[tag] = config
        for tag, polys in base_instance.polygons.items():
            target_instance.polygons[tag] = polys

    @abc.abstractmethod
    def generate_new(self):
        pass

class RoomConfig(object):
    def __init__(self):
        # config[tag] = {baseshape: Polygon, collisions: [bool], positions: [(x,y,yaw)]}
        self.config = dict()
        # polygons[tag] = [Polygon]
        self.polygons = dict()

        self.conf_tag_collisions = 'collisions'
        self.conf_tag_positions = 'positions'
        self.conf_tag_baseshape = 'baseshape'
    
    @abc.abstractmethod
    def prepare(self):
        pass

    def register(self, tag, base_shape, count):
        self.config[tag] = {
            self.conf_tag_collisions: np.ones([count], dtype=np.bool8),
            self.conf_tag_positions: np.zeros([count,3]),
            self.conf_tag_baseshape: base_shape,
        }
        self.polygons[tag] = np.empty([count], dtype=object)

    def tweak_collision(self, tag, index, val=True):
        self.config[tag][self.conf_tag_collisions][index] = val
    
    def set_config_collisions(self, tag, is_collision):
        self.config[tag][self.conf_tag_collisions] = np.array(is_collision, dtype=np.bool8)
    
    def set_config_positions(self, tag, x_y_yaw):
        self.config[tag][self.conf_tag_positions] = np.array(x_y_yaw)

    def set_config_baseshape(self, tag, base_shape):
        self.config[tag][self.conf_tag_baseshape] = base_shape

    def set_config_all(self, tag, is_collision, x_y_yaw, base_shape):
        self.set_config_collisions(tag, is_collision)
        self.set_config_positions(tag, x_y_yaw)
        self.set_config_baseshape(tag, base_shape)

    def set_config(self, tag, is_collision, x_y_yaw):
        self.set_config_collisions(tag, is_collision)
        self.set_config_positions(tag, x_y_yaw)

    def set_polygons_direct(self, tag, polygons):
        self.polygons[tag] = polygons

    def set_polygons_auto(self, tag):
        assert tag in self.config.keys(), 'not registered tag'
        base_shape = self.config[tag][self.conf_tag_baseshape]
        x_y_yaw = self.config[tag][self.conf_tag_positions]
        polys = [get_moved_poly_tf_rt(base_shape, p[0], p[1], p[2]) for p in x_y_yaw]
        self.set_polygons_direct(tag, polys)

    def get_collisions(self, component_tag):
        return self.config[component_tag][self.conf_tag_collisions]

    def get_positions(self, component_tag):
        return self.config[component_tag][self.conf_tag_positions]

    def get_baseshape(self, component_tag):
        return self.config[component_tag][self.conf_tag_baseshape]

    def get_polygons(self, component_tag):
        return self.polygons[component_tag]

    def gather_polygon_from_config(self, conf_filter=None):
        '''
        note: conf_filter require 3 args of [tag, config, index].
        '''
        l = []
        for tag, config in self.config.items():
            if conf_filter is None:
                l.extend(self.polygons[tag])
            else:
                polys = [p for i,p in enumerate(self.polygons[tag]) if conf_filter(tag, config, i)]
                l.extend(polys)
        return l

    def get_inner_poly(self, exterior_tag, holes=[], exterior_index=0):
        exte_pol = self.polygons[exterior_tag][exterior_index]
        assert len(exte_pol.interiors) == 1, 'exterior_tag should be perforated shape'
        inte_pol = Polygon(exte_pol.interiors[0])
        if isinstance(holes, Polygon):
            h = [holes.exterior.coords]
        elif isinstance(holes, MultiPolygon):
            h = [p.exterior.coords for p in holes.geoms]
        else:
            h = [p.exterior.coords for p in holes]
        return Polygon(inte_pol.exterior.coords, h)

    def get_space_poly(self, exterior_tag, poly_index=0):
        conf_filter = lambda tag, conf, i: (conf[self.conf_tag_collisions][i]) and (tag != exterior_tag)
        polys = unary_union(self.gather_polygon_from_config(conf_filter))
        return self.get_inner_poly(exterior_tag, polys, poly_index)

    def get_collision_poly(self):
        conf_filter = lambda tag, conf, i: conf[self.conf_tag_collisions][i]
        polys = self.gather_polygon_from_config(conf_filter)
        return unary_union(polys)

    @abc.abstractmethod
    def get_freespace_poly(self):
        pass

    @abc.abstractmethod
    def get_occupancy_grid(self, space_poly, origin_pos=(0,0), origin_ori=0, resolution=0.05, map_size=512, pass_color=255, obs_color=0):
        if (origin_pos[0] == 0) and (origin_pos[1] == 0) and (origin_ori == 0):
            corrected = space_poly
        else:
            corrected = get_moved_poly_rt_tf(space_poly, -origin_pos[0], -origin_pos[1], -origin_ori)
    
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