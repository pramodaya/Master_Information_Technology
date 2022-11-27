import Item
import Position2D
from Animal import Animal


class Cat(Animal):
    def __int__(self, position: Position2D, hunger_perc: float, items_destroyed: list[Item]):
        self.items_destroyed = items_destroyed
        super(Cat, self).__int__(position, hunger_perc)

    def move(self):
        hunger_p = super(Cat, self).get_hunger_perc() * 111 / 100
        super(Cat, self).set_hunger_perc(hunger_p)

    def meow(self) -> None:
        print("Dog meow")

    def destroy_item(self, item: Item):
        self.items_destroyed.remove(item)
