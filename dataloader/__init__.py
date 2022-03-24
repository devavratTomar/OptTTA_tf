import numpy as np
import albumentations as A

from scipy.ndimage import zoom

from utils import natural_sort
from .utils import *
import os
import random
import tensorflow as tf
from functools import partial


def get_dataloader(rootdir: str, sites: list, batch_size: int):
    """
    main function to get the data loader
    """
    img_paths = np.array(sorted([f for f in os.listdir(rootdir) if f.endswith('npy') and 'image' in f and check_sites(f, sites)]))
    seg_paths = np.array(sorted([f for f in os.listdir(rootdir) if f.endswith('npy') and 'mask' in f and check_sites(f, sites)]))
    assert len(img_paths) == len(seg_paths)

    n_total =  len(seg_paths)
    n_train = int(0.9 * n_total)

    print("Number of Images: %d" % len(img_paths))

    idxs = list(range(len(img_paths)))
    random.shuffle(idxs)

    train_img_paths = img_paths[idxs[:n_train]]
    train_seg_paths = seg_paths[idxs[:n_train]]

    val_img_paths = img_paths[idxs[n_train:]]
    val_seg_paths = seg_paths[idxs[n_train:]]

    # load numpy images in memory (we have lots, if not find other way)
    train_imgs = [np.load(os.path.join(rootdir, f)).astype(np.float32) for f in train_img_paths]
    train_segs = [np.load(os.path.join(rootdir, f)).astype(int) for f in train_seg_paths]

    val_imgs = [np.load(os.path.join(rootdir, f)).astype(np.float32) for f in val_img_paths]
    val_segs = [np.load(os.path.join(rootdir, f)).astype(int) for f in val_seg_paths]

    # tensorflow data loader
    train_dataset = tf.data.Dataset.from_tensor_slices({"image":train_imgs, "mask": train_segs})\
                                   .map(augment_train, num_parallel_calls=tf.data.AUTOTUNE)\
                                   .prefetch(tf.data.AUTOTUNE)\
                                   .map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)\
                                   .batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)\
                                   .prefetch(tf.data.AUTOTUNE)\
                                   .shuffle(buffer_size=n_total)
                                    
    val_dataset   = tf.data.Dataset.from_tensor_slices({"image":val_imgs, "mask":val_segs})\
                                   .map(augment_test, num_parallel_calls=tf.data.AUTOTUNE)\
                                   .prefetch(tf.data.AUTOTUNE)\
                                   .map(set_shapes, num_parallel_calls=tf.data.AUTOTUNE)\
                                   .batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)\
                                   .prefetch(tf.data.AUTOTUNE)\

    return train_dataset, val_dataset


### dont need tf dataloader as we load one volume at a time
class GenericNumpyVolumeLoader:
    def __init__(self, rootdir, sites):
        self.rootdir = rootdir
        imgs_paths = np.array(natural_sort([f for f in os.listdir(rootdir) if f.endswith('npy') and 'image' in f and check_sites(f, sites)]))
        seg_paths  = np.array(natural_sort([f for f in os.listdir(rootdir) if f.endswith('npy') and 'mask' in f and check_sites(f, sites)]))

        # get all patients
        patients = np.unique(["-".join(f.split("-")[:2]) for f in imgs_paths])
        print("All Patients: \n", patients)

        patients_volumes = []
        patients_segs    = []

        for p in patients:
            slices = [np.load(os.path.join(self.rootdir, f)).astype(np.float32) for f in imgs_paths if p in f]
            segs   = [np.load(os.path.join(self.rootdir, f)).astype(np.int32) for f in seg_paths if p in f]

            slices = np.stack(slices, axis=0)
            segs   = np.stack(segs, axis=0)

            # shape to 256 256
            h, w = slices.shape[1:]
            slices = zoom(slices, (1, 256/h, 256/w), order=2)
            segs   = zoom(segs, (1, 256/h, 256/w), order=0 )

            slices = slices[..., np.newaxis] # batch h w ch
            slices = 2 * slices - 1          # normalize to -1 and 1

            patients_volumes.append(slices)
            patients_segs.append(segs)

        self.patients_volumes = patients_volumes
        self.patients_segs    = patients_segs
        self.patient_names = patients

    def __getitem__(self, index):
        return self.patients_volumes[index], self.patients_segs[index]

    def __len__(self):
        return len(self.patients_volumes)
