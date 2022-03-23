from matplotlib import image
import tensorflow as tf
import numpy as np
import math
import random
from tensorflow.keras import layers

################################################ Appearance Transformations ###################################
class Indentity(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, x):
        return x

class GaussainBlur(tf.keras.Model):
    def __init__(self, n_channels=1, kernel_size=7):
        super().__init__()
        self.padding = kernel_size//2
        # non leanable parameter
        self.base_gauss = tf.constant(self.get_gaussian_kernel2d(kernel_size, n_channels), dtype=tf.float32)

        # learnable parameter
        self.sigma = tf.Variable(1.0, dtype=tf.float32)

    def gaussain_window(self, window_size):
        def gauss_fcn(x):
            return - (x - window_size//2)**2 / 2.0
        
        gauss = np.stack([math.exp(gauss_fcn(x)) for x in range(window_size)])
        return gauss
    
    def get_gaussian_kernel(self, ksize):
        window_1d = self.gaussain_window(ksize)
        return window_1d

    def get_gaussian_kernel2d(self, ksize, n_channels):
        kernel_x = self.get_gaussian_kernel(ksize)
        kernel_y = self.get_gaussian_kernel(ksize)

        kernel_2d = np.matmul(kernel_x[..., np.newaxis], kernel_y[..., np.newaxis].T)
        kernel_2d = kernel_2d[..., np.newaxis]
        kernel_2d = np.tile(kernel_2d, (1, 1, n_channels))[..., np.newaxis]
        kernel_2d = kernel_2d.astype(np.float32)
        return kernel_2d

    def __call__(self, x):
        gauss_kernel = tf.pow(self.base_gauss, 1.0/tf.square(self.sigma))
        gauss_kernel = gauss_kernel / gauss_kernel.sum() # lucky to have only one channel image, otherwise divide channel wise sum
        x = tf.pad(x,
                  tf.constant([[0,                       0],
                               [self.padding, self.padding],
                               [self.padding, self.padding],
                               [0,                       0]]), "REFLECT")
        
        x = tf.nn.depthwise_conv2d(x, gauss_kernel, [1, 1, 1, 1], 'VALID')
        return x



class CommonStyle(tf.keras.Model):
    def __init__(self, n_channels=1):
        super().__init__()
        self.bias = tf.Variable(np.zeros((1, 1, 1, n_channels)), dtype=tf.float32)
        self.range = tf.Variable(1e-3 * np.ones(1, 1, 1, n_channels), dtype=tf.float32)
        
        self.n_channels = n_channels

        self.uniform_dist = lambda size : tf.random.uniform(size, minval=-1.0, maxval=1.0, dtype=tf.float32)

    def denormalize(self, x):
        return 0.5 * x + 0.5

    def normalize(self, x):
        return 2.0 * x - 1.0

    def abs(self, x):
        return tf.nn.relu(x) + tf.nn.relu(-x)


class Brightness(CommonStyle):
    def __init__(self, n_channels=1):
        super().__init__(n_channels=n_channels)
    
    def __call__(self, x):
        size = tf.stack([tf.shape(x)[0], 1, 1, self.n_channels])
        x = self.denormalize(x)

        random_brightness = self.bias + self.abs(self.range) * self.uniform_dist(size)
        x = x + random_brightness

        x = self.normalize(x)
        return x


class Contrast(CommonStyle):
    def __init__(self, n_channels=1):
        super().__init__(n_channels=n_channels)

    def __call__(self, x):
        size = tf.stack([tf.shape(x)[0], 1, 1, self.n_channels])
        x = self.denormalize(x)

        random_contrast = 1.0 + self.bias + self.abs(self.range) * self.uniform_dist(size)
        x = x * random_contrast

        x = self.normalize(x)
        return x


class Gamma(CommonStyle):
    def __init__(self, n_channels=1):
        super().__init__(n_channels=1)

    def __call__(self, x):
        size = tf.stack([tf.shape(x)[0], 1, 1, self.n_channels])
        x = self.denormalize(x)

        random_gamma = 1.0 + self.bias + 1e-2 * self.abs(self.range) * self.uniform_dist(size)


################################################ Spatial Transformations ############################################
############# Helpers ################
def affine_grid_generator(height, width, theta):
    """
    This function returns a sampling grid, which when
    used with the bilinear sampler on the input feature
    map, will create an output feature map that is an
    affine transformation [1] of the input feature map.
    Input
    -----
    - height: desired height of grid/output. Used
      to downsample or upsample.
    - width: desired width of grid/output. Used
      to downsample or upsample.
    - theta: affine transform matrices of shape (num_batch, 2, 3).
      For each image in the batch, we have 6 theta parameters of
      the form (2x3) that define the affine transformation T.
    Returns
    -------
    - normalized grid (-1, 1) of shape (num_batch, 2, H, W).
      The 2nd dimension has 2 components: (x, y) which are the
      sampling points of the original image for each point in the
      target image.
    Note
    ----
    [1]: the affine transformation allows cropping, translation,
         and isotropic scaling.
    """
    num_batch = tf.shape(theta)[0]

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tf.cast(theta, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    batch_grids = tf.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids


def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.cast(tf.gather_nd(img, indices), dtype=tf.float32)


def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out


class RandomSpatial(tf.keras.Model):
    def __init__(self):
        super().__init__()

        ## non trainable params
        self.unit_affine = tf.constant(np.array([1, 0, 0, 0, 1, 0]).reshape(-1, 2, 3), dtype=tf.float32)

        ## register trainable parameters as required
        self.register_custom_parameters()

        ## unifrom distribution gettter
        self.uniform_dist = lambda size : tf.random.uniform(size, minval=-1.0, maxval=1.0, dtype=tf.float32)

    def register_custom_parameters(self):
        raise NotImplementedError

    def generate_random_affine(self, batch_size):
        raise NotImplementedError

    def __call__(self, x):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]

        random_affine = self.generate_random_affine(B)
        
        grid = affine_grid_generator(H, W, random_affine)
        grid_x, grid_y = grid[:, 0, :, :], grid[:, 1, :, :]

        x = bilinear_sampler(x, grid_x, grid_y)
        return x


    def test(self, x, random_affine=None):
        B = tf.shape(x)[0]
        H = tf.shape(x)[1]
        W = tf.shape(x)[2]
        
        if random_affine is None:
            random_affine = self.generate_random_affine(B)

        grid = affine_grid_generator(H, W, random_affine)
        grid_x, grid_y = grid[:, 0, :, :], grid[:, 1, :, :]
        
        x = bilinear_sampler(x, grid_x, grid_y)

        return x, random_affine

    def get_homographic_mat(self, A):
        ## A is batch x 2 x 3 matrix
        H = tf.pad(A, tf.constant([[0, 0],
                                   [0, 1],
                                   [0, 0]]), mode="CONSTANT")
        # H is now batch, 3, 3, but all values are zero
        H[..., -1, -1] += 1.0 # add 1 to the last row and last column

        return H

    def invert_affine(self, affine):
        homo_affine = self.get_homographic_mat(affine)
        inv_homo_affine = tf.linalg.inv(homo_affine)
        inv_affine = inv_affine[:, :2, :3]

        return inv_affine

    def abs(self, x):
        return tf.nn.relu(x) + tf.nn.relu(-x)


class RandomResizeCrop(RandomSpatial):
    def __init__(self):
        super().__init__()

    def register_custom_parameters(self):
        self.delta_scale_x = tf.Variable(0.1, dtype=tf.float32)
        self.delta_scale_y = tf.Variable(0.1, dtype=tf.float32)

        self.scale_matrix_x = tf.constant(np.array([1, 0, 0, 0, 0, 0]).reshape(-1, 2, 3), dtype=tf.float32)
        self.scale_matrix_y = tf.constant(np.array([0, 0, 0, 0, 1, 0]).reshape(-1, 2, 3), dtype=tf.float32)
        self.translation_matrix_x = tf.constant(np.array([0, 0, 1, 0, 0, 0]).reshape(-1, 2, 3), dtype=tf.float32)
        self.translation_matrix_y = tf.constant(np.array([0, 0, 0, 0, 0, 1]).reshape(-1, 2, 3), dtype=tf.float32)

    def get_random(self, batch_size):
        return tf.random.uniform(tf.stack([batch_size, 1, 1]), minval=-1, maxval=1)

    def generate_random_affine(self, batch_size):
        delta_x = 0.5 * self.delta_scale_x * self.get_random(batch_size)
        delta_y = 0.5 * self.delta_scale_y * self.get_random(batch_size)

        affine = (1 - self.abs(self.delta_scale_x)) * self.scale_matrix_x +\
                 (1 - self.abs(self.delta_scale_y)) * self.scale_matrix_y +\
                 delta_x * self.translation_matrix_x + delta_y * self.translation_matrix_y
        return affine


class RandomHorizontalFlip(RandomSpatial):
    def __init__(self):
        super().__init__()

    def register_custom_parameters(self):
        self.horizontal_flip = tf.constant(np.array([-1, 0, 0, 0, 1, 0]).reshape(-1, 2, 3), dtype=tf.float32)

    def generate_random_affine(self, batch_size):
        affine = tf.tile(self.unit_affine, tf.stack([batch_size, 1, 1]))

        # randomly flip some of the images in the batch
        mask = tf.random.uniform(batch_size) > 0.5
        affine[mask] = affine[mask] * self.horizontal_flip

        return affine


class RandomVerticalFlip(RandomSpatial):
    def __init__(self):
        super().__init__()

    def register_custom_parameters(self):
        self.horizontal_flip = tf.constant(np.array([1, 0, 0, 0, -1, 0]).reshape(-1, 2, 3), dtype=tf.float32)

    def generate_random_affine(self, batch_size):
        affine = tf.tile(self.unit_affine, tf.stack([batch_size, 1, 1]))

        # randomly flip some of the images in the batch
        mask = tf.random.uniform(batch_size) > 0.5
        affine[mask] = affine[mask] * self.horizontal_flip

        return affine


class RandomRotate(RandomSpatial):
    def __init__(self):
        super().__init__()

    def register_custom_parameters(self):
        self.horizontal_flip = tf.constant(np.array([0, -1, 0, 1, 0, 0]).reshape(-1, 2, 3), dtype=tf.float32)

    def generate_random_affine(self, batch_size):
        affine = tf.tile(self.unit_affine, tf.stack([batch_size, 1, 1]))

        # randomly flip some of the images in the batch
        mask = tf.random.uniform(batch_size) > 0.5
        affine[mask] = affine[mask] * self.horizontal_flip

        return affine


class DummyAugmentor(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x