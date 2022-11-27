import Position2D


class MovableObject:
    def __init__(self, position: Position2D):
        self.position = position

    def move(self):
        print("Movable object move")


