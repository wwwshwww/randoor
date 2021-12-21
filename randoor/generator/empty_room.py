from shapely.geometry.polygon import Polygon

from ..spawner.poly import random_triangulation
from ..generator_factory import RoomConfig, RoomGeneratorFactory

class EmptyRoomConfig(RoomConfig):
    def __init__(self, wall_shape):
        super(EmptyRoomConfig, self).__init__()
        self.wall_shape = wall_shape

        self.wall_interior_polygon = Polygon(self.wall_shape.interiors[0])
        self.wall_exterior_polygon = None

        self.tag_wall = 'wall'

    def prepare(self):
        self.register(self.tag_wall, self.wall_shape, 1)

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
        p = random_triangulation(n_points, x_min, x_max, y_min, y_max)
        return p.exterior.buffer(self.room_wall_thickness, join_style=2) # mitre style

    def generate_new(self):
        wall_shape = self._create_wall_poly()
        wall_collision = [True]
        wall_pos = [(0,0,0)]

        emp_room = EmptyRoomConfig(
            wall_shape=wall_shape
        )

        emp_room.prepare()
        emp_room.set_config(emp_room.tag_wall, wall_collision, wall_pos)
        emp_room.set_polygons_auto(emp_room.tag_wall)

        return emp_room