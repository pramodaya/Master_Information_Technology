import Position2D
from MovableObject import MovableObject


class Animal(MovableObject):
    def __init__(self, position: Position2D, hunger_perc: float):
        self.hunger_perc = hunger_perc
        super(Animal, self).__init__(position)

    def get_hunger_perc(self) -> float:
        return self.hunger_perc

    def set_hunger_perc(self, hunger_perc):
        self.hunger_perc = hunger_perc

    def eat(self) -> None:
        self.hunger_perc = self.hunger_perc * 90 / 100

    def sleep(self, hours: float) -> None:
        self.hunger_perc = self.hunger_perc + (self.hunger_perc * 10 / 100) * hours
