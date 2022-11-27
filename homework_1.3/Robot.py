import Position2D
from MovableObject import MovableObject


class Robot(MovableObject):
    def __init__(self, position: Position2D, battery_perc: int):
        self.battery_perc = battery_perc
        super(Robot, self).__init__(position)

    def charge(self, hours: float):
        self.battery_perc = self.battery_perc + (self.battery_perc * 10 / 100) * hours

    def move(self) -> None:
        self.battery_perc = self.battery_perc * 90 / 100
