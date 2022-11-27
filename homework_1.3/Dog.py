import Position2D
from Animal import Animal


class Dog(Animal):
    def __int__(self, position: Position2D, hunger_perc: float, bones_hidden: int):
        self.bones_hidden = self.bones_hidden
        super(Dog, self).__int__(position, hunger_perc)

    def move(self) -> None:
        hunger_p = super(Dog, self).get_hunger_perc() + super(Dog, self).get_hunger_perc() * 10 / 100
        super(Dog, self).set_hunger_perc(hunger_p)

    def bark(self) -> None:
        print("Dog bark")
