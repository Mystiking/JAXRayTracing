import jax.numpy as jnp
from Vec3 import Vec3
from functools import reduce


class Sphere:
    '''
    Class for a sphere defined as a center and a radius.
    '''
    def __init__(self,
                 center: Vec3,
                 radius: float,
                 diffuse: Vec3):
        self.center = center
        self.radius = radius
        self.diffuse = diffuse  # The diffuse color of the material

    def intersect(self, origin, direction, far=1.0e39):
        '''
        Determines where the sphere was hit by a ray.
        This can be done by solving the problem:
            |origin - t*direction - sphere.center|^2 - sphere.radius^2 = 0
        which is a 2nd degree polynomial on the form
            f(x) = ax^2 + bx + c
        where x = t and,
            a = direction^2 = 1 (direction is normalized)
            b = 2 * direction * (origin - sphere.center)
            c = |origin - sphere.center|^2 - R^2
        the solution to this problem is of course given by,
            x = (-b +- sqrt(b^2 - 4ac)) / 2a
        If we denote b^2 - 4ac as d, we have
            0 intersections when d < 0
            1 intersection when d = 0
            2 intersections when d > 0

        '''
        b = 2. * direction.dot(origin - self.center)
        c = abs(self.center) + abs(origin) - 2 * self.center.dot(origin) - (self.radius * self.radius)
        d = b**2 - 4*c
        dsq = jnp.sqrt(jnp.maximum(0, d))

        x0 = (-b - dsq) / (2.)
        x1 = (-b + dsq) / (2.)

        x = jnp.where((x0 > 0) & (x0 < x1), x0, x1)

        hit = (d > 0) & (x > 0)
        return jnp.where(hit, x, far)  # Blow every point very far away

    def diffusecolor(self, rayhit):
        return self.diffuse

    def light(self, origin, direction, intersection, light_position, eye_position, scene_objects, bounce = 0, far=1.0e15):
        '''
        Basic light model using a only diffuse lighting
        '''
        rayhit = origin + direction * intersection
        normal = ((rayhit - self.center) * (1. / self.radius))
        direction_to_light = (light_position - rayhit).norm()
        direction_to_eye = (eye_position - rayhit).norm()
        nudged = rayhit + normal * 0.001  # To avoid shadow acne

        # Create shadow mask
        light_distances = [o.intersect(nudged, direction_to_light, far=far) for o in scene_objects]
        light_nearest = reduce(jnp.minimum, light_distances)
        light_mask = light_distances[scene_objects.index(self)] == light_nearest

        # Ambient light
        color = Vec3(0.05, 0.05, 0.05)

        # Lambert shading (diffuse)
        light_hit = jnp.maximum(normal.dot(direction_to_light), 0)
        color += self.diffusecolor(rayhit) * light_hit * light_mask

        # Phong light
        phong = normal.dot((direction_to_light + direction_to_eye).norm())
        color += Vec3(1., 1., 1.) * jnp.power(jnp.clip(phong, 0, 1), 50) * light_mask

        return color
