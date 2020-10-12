import jax.numpy as jnp
import jax
from functools import reduce
from Camera import Camera
from Ray import Ray
from Vec3 import Vec3
from Scene import Scene
from Shapes import Sphere


class Renderer:
    '''
    Class responsible for rendering an image.
    '''
    def __init__(self, camera: Camera, scene: Scene, light_position: Vec3 = Vec3(5., 5., -10.)):
        self.camera = camera
        self.scene = scene
        self.light_position = light_position

    def ray_color(self, r: Ray):
        '''
        Evaluates the color of a ray based on where it hits the background.
        '''
        unit_direction = r.ray_direction / jnp.linalg.norm(r.ray_direction)
        t = 0.5 * (unit_direction[1] + 1.0)
        return (1.0-t) * jnp.array([1., 1., 1.]) + t * jnp.array([0.5, 0.7, 1.0])

    def render(self):
        image = jnp.zeros((self.camera.height, self.camera.width, 3))
        for j in range(self.camera.height-1, -1, -1):
            for i in range(self.camera.width):
                u = float(i) / (self.camera.width - 1)
                v = float(j) / (self.camera.height - 1)
                r = Ray(self.camera.origin,
                        self.camera.lower_left_corner + u*self.camera.horizontal + v*self.camera.vertical - self.camera.origin)
                image = jax.ops.index_update(image, jax.ops.index[j, i, :], self.ray_color(r))
        return image

    def ray_trace(self, origin, normalized_direction):
        far = 1.0e15  # A large number, which we can never hit
        distances = [o.intersect(origin, normalized_direction, far) for o in self.scene.objects]
        nearest = reduce(jnp.minimum, distances)
        color = Vec3(0, 0, 0)
        for (o, d) in zip(self.scene.objects, distances):
            color += o.light(origin, normalized_direction, d, self.light_position, origin, self.scene.objects) * (nearest != far) * (d == nearest)
        return color

    def render_fast(self):
        r = float(self.camera.width) / self.camera.height
        S = (-1., 1. / r + .25, 1., -1. / r + .25)
        x = jnp.tile(jnp.linspace(S[0], S[2], self.camera.width), self.camera.height)
        y = jnp.repeat(jnp.linspace(S[1], S[3], self.camera.height), self.camera.width)
        origin = Vec3(self.camera.origin[0], self.camera.origin[1], self.camera.origin[2])
        image = self.ray_trace(origin, (Vec3(x, y, 0) - origin).norm())
        return image


if __name__ == '__main__':
    cam = Camera(width=400, origin=jnp.array([0, 0.05, -1.]))
    num_spheres = 50
    scene_objects = []
    min_x = -7.5
    min_y = .5
    min_z = 2.5
    size_deviation = 0.5
    max_deviation_x = 15.
    max_deviation_y = 10.
    max_deviation_z = 20.
    import random
    colorful_factor = 0.35
    for i in range(1, num_spheres+1):
        color = Vec3(random.random(), random.random(), random.random())
        color_to_add = random.randint(0, 2)
        if color_to_add == 0:
            color.x += colorful_factor
            color.y -= colorful_factor
            color.z -= colorful_factor
        elif color_to_add == 1:
            color.x -= colorful_factor
            color.y += colorful_factor
            color.x -= colorful_factor
        else:
            color.x -= colorful_factor
            color.y -= colorful_factor
            color.x += colorful_factor

        x_coord = random.random() * max_deviation_x + min_x
        y_coord = random.random() * max_deviation_y + min_y
        z_coord = random.random() * max_deviation_z + min_z
        scene_objects.append(
            Sphere(Vec3(x_coord, y_coord, z_coord), .6 + size_deviation * random.random(), color)
        )
    scene_objects.append(
        Sphere(Vec3(0., -99999.5, 0.), 99999, Vec3(0.75, 0.75, 0.75))
    )
    scene = Scene(objects=scene_objects)
    renderer = Renderer(cam, scene)
    image_data = renderer.render_fast()

    red_channel = 255 * jnp.clip(image_data.x, 0, 1).reshape((cam.height, cam.width))
    green_channel = 255 * jnp.clip(image_data.y, 0, 1).reshape((cam.height, cam.width))
    blue_channel = 255 * jnp.clip(image_data.z, 0, 1).reshape((cam.height, cam.width))

    # As we're writing with openCV, the red and blue channel are swapped
    image = jnp.stack([blue_channel, green_channel, red_channel], axis=-1)

    # Create an actual image from the Vec3 object
    import cv2
    import numpy as np
    cv2.imwrite("image.png", np.array(jnp.asarray(image)))
