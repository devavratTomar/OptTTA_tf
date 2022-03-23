import albumentations as A
import tensorflow as tf
import numpy as np

TRAIN_TRANSFORM = A.Compose([A.HorizontalFlip(),
                             A.VerticalFlip(),
                             A.Rotate(p=0.5),
                             A.GaussianBlur(),
                             A.RandomBrightness(0.2),
                             A.RandomContrast(0.2),
                             A.RandomGamma(),
                             A.RandomResizedCrop(256, 256, scale=(0.5, 1.0))])

TEST_TRANSFORM = A.Resize(256, 256)


def check_sites(f, sites):
    for site in sites:
        if site in f:
            return True
    return False


def train_aug_fn(img, seg):
    aug_data = TRAIN_TRANSFORM(image=img, mask=seg)
    aug_img, aug_seg = aug_data["image"], aug_data["mask"]
    aug_img = tf.cast(aug_img, tf.float32)
    aug_seg = tf.cast(aug_seg, tf.int32)

    # -1 to 1
    aug_img = 2*aug_img - 1
    aug_img = aug_img[..., np.newaxis] # add channel

    return aug_img, aug_seg

def test_aug_fn(img, seg):
    aug_data = TEST_TRANSFORM(image=img, mask=seg)
    aug_img, aug_seg = aug_data["image"], aug_data["mask"]
    aug_img = tf.cast(aug_img, tf.float32)
    aug_seg = tf.cast(aug_seg, tf.int32)

    # -1 to 1
    aug_img = 2*aug_img - 1
    aug_img = aug_img[..., np.newaxis] # add channel

    return aug_img, aug_seg

def augment_train(data):
    img, seg = data["image"], data["mask"]
    aug_img, aug_seg = tf.numpy_function(func=train_aug_fn, inp=[img, seg], Tout=(tf.float32, tf.int32))
    return aug_img, aug_seg

def augment_test(data):
    img, seg = data["image"], data["mask"]
    aug_img, aug_seg = tf.numpy_function(func=test_aug_fn, inp=[img, seg], Tout=(tf.float32, tf.int32))
    return aug_img, aug_seg

def set_shapes(img, seg):
    # img, seg = data
    img.set_shape((256, 256, 1))
    seg.set_shape((256, 256))
    return img, seg