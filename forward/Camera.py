import jax.numpy as jnp


class Camera:
    '''
    A class for keeping track of camera parameters.
    '''
    def __init__(self,
                 aspect_ratio: float = 16.0 / 9.0,
                 width: int = 400,
                 viewport_height: float = 2.0,
                 focal_length: float = 1.0,
                 origin: jnp.ndarray = jnp.array([0., 0., 0.])):
        self.aspect_ratio = aspect_ratio
        self.width = width
        self.height = int(width / aspect_ratio)
        self.viewport_height = viewport_height
        self.viewport_width = aspect_ratio * viewport_height
        self.focal_length = focal_length

        self.origin = origin
        self.horizontal = jnp.array([self.viewport_width, 0., 0.])
        self.vertical = jnp.array([0, self.viewport_height, 0.])
        self.lower_left_corner = origin - self.horizontal/2 - self.vertical/2 - jnp.array([0, 0, focal_length])
