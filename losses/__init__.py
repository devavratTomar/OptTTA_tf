import tensorflow as tf
import tensorflow.keras.layers as layers


def compute_loss(model, image, label, is_train=True):
    logits = model.call(image, is_train)
    label_bin = tf.clip_by_value(label, 0, 3)
    label_vec = tf.one_hot(tf.cast(label_bin, tf.int64), depth=model.c, dtype=tf.int64)
    # print(label_vec.shape)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_vec, logits=logits))
    return loss


def compute_dice_loss(model, image, label, is_train=True):
    logits = model.call(image, is_train)
    softmax = tf.nn.softmax(logits)
    label_bin = tf.clip_by_value(label, 0, 3)
    label_vec = tf.one_hot(tf.cast(label_bin, tf.int64), depth=model.c, dtype=tf.float32)
    # print(label_vec.shape)ss
    overlap = tf.reduce_sum(softmax * label_vec, axis=(1, 2, 3))
    union = tf.reduce_sum(softmax + label_vec, axis=(1, 2, 3))
    dice = tf.reduce_mean(2 * overlap / (union + 1e-7))
    return 1 - dice


def compute_ce_dice_loss(model, image, label, is_train=True):
    logits = model.call(image, is_train)
    label_bin = tf.clip_by_value(label, 0, 3)
    label_vec = tf.one_hot(tf.cast(label_bin, tf.int64), depth=model.c, dtype=tf.int64)
    # print(label_vec.shape)
    celoss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_vec, logits=logits))
    softmax = tf.nn.softmax(logits)
    overlap = tf.reduce_sum(softmax * tf.cast(label_vec, tf.float32), axis=(1, 2, 3))
    union = tf.reduce_sum(softmax + tf.cast(label_vec, tf.float32), axis=(1, 2, 3))
    dice = tf.reduce_mean(2 * overlap / (union + 1e-7))
    loss = celoss + 1 - dice
    return loss


def compute_weighted_loss(model, image, label,
                          ce=tf.keras.losses.CategoricalCrossentropy(from_logits=True,
                                                                     reduction=tf.keras.losses.Reduction.NONE),
                          is_train=True):
    logits = model(image, training=is_train)
    # label_bin = tf.clip_by_value(label, 0, 3)
    label_vec = tf.one_hot(tf.cast(label, tf.int64), depth=logits.shape[-1], dtype=tf.int64)
    weights = tf.cast(1.0 - tf.reduce_sum(label_vec, axis=(0, 1, 2), keepdims=True) / tf.reduce_sum(label_vec), tf.float32)
    # print(label_vec.shape)
    celoss = ce(label_vec, logits)
    # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_vec, logits=logits))
    celoss = tf.cast(label_vec, tf.float32) * weights * tf.cast(tf.expand_dims(celoss, axis=-1), tf.float32)
    celoss = tf.reduce_sum(celoss) / (tf.reduce_sum(weights) * label.shape[0] * label.shape[-1])
    softmax = tf.nn.softmax(logits)
    overlap = tf.reduce_sum(softmax * tf.cast(label_vec, tf.float32), axis=(1, 2, 3))
    union = tf.reduce_sum(softmax + tf.cast(label_vec, tf.float32), axis=(1, 2, 3))
    dice = tf.reduce_mean(2 * overlap / (union + 1e-7))
    loss = celoss + 1 - dice
    return loss


class NuclearNorm(layers.Layer):
    def __init__(self, poolsize=4):
        super(NuclearNorm, self).__init__()
        self.pool_size = poolsize
        # self.tconv = layers.Conv2DTranspose(self.filter, kernel_size=2, padding='same', activation='relu', strides=2)
        # self.out_shape = tf.shape()

    def call(self, x):  # the prediction logits you need to apply softmax to get the propability
        x = tf.nn.softmax(x)
        x = layers.MaxPool2D(pool_size=self.pool_size)(x)  # batch x h x w x n_classes
        x = tf.reshape(x, shape=[tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2],
                                 tf.shape(x)[3]])  # batch x hw x n_classes
        nn = tf.reduce_sum(tf.linalg.svd(x, compute_uv=False), axis=-1)
        loss = tf.reduce_mean(nn)
        return loss


class CE_loss(layers.Layer):
    def __init__(self, model, num_c=3):
        super(CE_loss, self).__init__()
        self.num_c = num_c
        self.model = model

    def call(self, x):
        x = self.model.call(x)
        p = tf.nn.softmax(x)
        loss = tf.reduce_mean(tf.reduce_sum(-p * tf.math.log(p + 1e-6), axis=-1))
        return loss


class Cosine_Similarity(layers.Layer):
    def __init__(self):
        super(Cosine_Similarity, self).__init__()

    def call(self, img, ref):
        img = tf.cast(img, dtype=tf.float32)
        ref = tf.cast(ref, dtype=tf.float32)
        img_norm = tf.norm(img, ord='euclidean', axis=None, keepdims=None, name=None)
        ref_norm = tf.norm(ref, ord='euclidean', axis=None, keepdims=None, name=None)
        img_mult_ref = tf.reduce_sum(tf.multiply(img, ref))
        cos_sim = img_mult_ref / (img_norm * ref_norm + 1e-7)
        return cos_sim


class Smooth_loss(layers.Layer):
    def __init__(self, unet, num_c=3, poolsize=4):
        super(Smooth_loss, self).__init__()
        self.num_c = num_c

        self.CE_loss = CE_loss(self.model, self.num_c)
        self.pool_size = poolsize
        self.NuclearNorm = NuclearNorm(self.pool_size)
        self.CosineSim = Cosine_Similarity()

    def call(self, unet, x, alpha):  # only 1 parameter since we don't use batch norm
        loss_ce = self.CE_loss.call(x) * 100
        loss_nn = alpha * tf.math.log(self.NuclearNorm(x) + 1e-7) * self.NuclearNorm(x)
        loss = loss_ce + loss_nn
        return loss


def cos_sim(img, ref):
    img = tf.cast(img, dtype=tf.float32)
    ref = tf.cast(ref, dtype=tf.float32)
    img_norm = tf.norm(img, ord='euclidean', axis=None, keepdims=None, name=None)
    ref_norm = tf.norm(ref, ord='euclidean', axis=None, keepdims=None, name=None)
    img_mult_ref = tf.reduce_sum(tf.multiply(img, ref))
    cos_sim = img_mult_ref / (img_norm * ref_norm + 1e-7)
    return cos_sim
