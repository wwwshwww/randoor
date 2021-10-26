import numpy as np
import shapely
from shapely.geometry.polygon import Polygon

from ..spawner import poly
from ..generator_factory import RoomConfig, RoomGeneratorFactory

class EmptyRoomConfig(RoomConfig):
    def __init__(self, wall_shape, wall_thickness):
        super(EmptyRoomConfig, self).__init__()
        self.wall_shape = wall_shape
        self.wall_thickness = wall_thickness

        self.wall_interior_polygon = Polygon(self.wall_shape.interiors[0])
        self.wall_exterior_polygon = None

        self.tag_wall = 'wall'
        self.prepare()

    @property
    def wall_pose(self):
        return self.config[self.tag_wall][self.conf_tag_positions]

    def prepare(self):
        self.register_config(self.tag_wall, 1)

    def get_freespace_poly(self):
        return self.get_space_poly(self.tag_wall)

class EmptyRoomGenerator(RoomGeneratorFactory):
    def __init__(self, 
                 room_length_max=9,
                 room_wall_thickness=0.05):
        
        super(EmptyRoomGenerator, self).__init__()
        self.room_length_max = room_length_max
        self.room_wall_thickness = room_wall_thickness

    def _create_wall_poly(self):
        n_points = self.room_length_max * 6
        x_max = self.room_length_max / 2
        x_min = -x_max
        y_max = x_max
        y_min = -x_max
        p = poly.random_triangulation(n_points, x_min, x_max, y_min, y_max)
        return p.exterior.buffer(self.room_wall_thickness, join_style=2) # mitre style

    def generate_new(self):
        wall_shape = self._create_wall_poly()
        emp_room = EmptyRoomConfig(
            wall_shape=wall_shape,
            wall_thickness=self.room_wall_thickness
        )
        wall_collision = True
        wall_pos = [(0,0,0)]
        wall_poly = [wall_shape]
        emp_room.set_config(emp_room.tag_wall, collision=True, positions=(0,0,0), polygons=wall_poly)

        return emp_room