import jax.numpy as jnp
from jax import grad
from Render import Renderer
from Scene import Scene
from Shapes import Sphere
from Camera import Camera
from Vec3 import Vec3

# Constructing the ground-truth
cam = Camera(width=400, origin=jnp.array([0, 0.05, -1.]))
gt_object = Sphere(Vec3(0., 0.5, .75), .6, Vec3(0., 0., 1.))
scene = Scene(objects=[gt_object])
renderer = Renderer(cam, scene)

image_data = renderer.render_fast()
print(type(image_data.y))

red_channel = 255 * jnp.clip(image_data.x, 0, 1).reshape((cam.height, cam.width))
green_channel = 255 * jnp.clip(image_data.y, 0, 1).reshape((cam.height, cam.width))
blue_channel = 255 * jnp.clip(image_data.z, 0, 1).reshape((cam.height, cam.width))

gt_image = jnp.stack([blue_channel, green_channel, red_channel], axis=-1)

import cv2
import numpy as np
import os
if not os.path.exists("out"):
    os.mkdir("out")

cv2.imwrite("image_target.png", np.array(jnp.asarray(gt_image)))

def save_image(color, i):
    _object = Sphere(Vec3(0., 0.5, .75), .6, Vec3(color[0], color[1], color[2]))
    renderer.scene.objects = [_object]
    image_data = renderer.render_fast()

    red_channel = 255 * jnp.clip(image_data.x, 0, 1).reshape((cam.height, cam.width))
    green_channel = 255 * jnp.clip(image_data.y, 0, 1).reshape((cam.height, cam.width))
    blue_channel = 255 * jnp.clip(image_data.z, 0, 1).reshape((cam.height, cam.width))

    image = jnp.stack([blue_channel, green_channel, red_channel], axis=-1)
    cv2.imwrite("out/image_{0}.png".format(i), np.array(jnp.asarray(image)))

# Constructing the loss function of the optimization
def loss(color):
    _object = Sphere(Vec3(0., 0.5, .75), .6, Vec3(color[0], color[1], color[2]))
    renderer.scene.objects = [_object]
    image_data = renderer.render_fast()

    red_channel = 255 * jnp.clip(image_data.x, 0, 1).reshape((cam.height, cam.width))
    green_channel = 255 * jnp.clip(image_data.y, 0, 1).reshape((cam.height, cam.width))
    blue_channel = 255 * jnp.clip(image_data.z, 0, 1).reshape((cam.height, cam.width))

    image = jnp.stack([blue_channel, green_channel, red_channel], axis=-1)
    return jnp.linalg.norm(image - gt_image)

dloss = grad(loss)

starting_color = jnp.array([1., 0., 0.])
train_iters = 100
lr = 1e-6
for i in range(train_iters):
    save_image(starting_color, i)
    print("iter={0}, loss={1:.3}".format(i+1, loss(starting_color)))
    dc = dloss(starting_color)
    starting_color -= lr * dc

save_image(starting_color, "final")
print(starting_color)
