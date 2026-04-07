class Point3D :
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0
        self.value = 0.0

    def toString(self):
        return f"{self.x} {self.y} {self.z} {self.value}"
