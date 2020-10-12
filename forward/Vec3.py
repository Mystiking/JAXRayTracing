import jax.numpy as jnp

class Vec3():
    def __init__(self, x, y, z):
        (self.x, self.y, self.z) = (x, y, z)

    def __mul__(self, other):
        return Vec3(self.x * other, self.y * other, self.z * other)

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def dot(self, other):
        return (self.x * other.x) + (self.y * other.y) + (self.z * other.z)

    def __abs__(self):
        return self.dot(self)

    def norm(self):
        mag = jnp.sqrt(abs(self))
        return self * (1.0 / jnp.where(mag == 0, 1, mag))

    def components(self):
        return (self.x, self.y, self.z)
