import tensorflow as tf
from tensorflow.keras import layers


class DoubleConv(tf.keras.layers.Layer):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.double_conv = tf.keras.Sequential([
            layers.Conv2D(mid_channels, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(out_channels, kernel_size=3, padding="same"),
            layers.BatchNormalization(),
            layers.ReLU(),
        ])

    def call(self, x, training=False, exp_stat=False):
        if exp_stat:
            pre_bn = []
            for layer in self.double_conv.layers:
                x = layer(x, training=training)
                if 'conv2d' in layer.name and 'input' not in layer.name:
                    pre_bn.append(x)
            return x, pre_bn
        return self.double_conv(x, training=training)


class Down(tf.keras.layers.Layer):
    """Downscaling with maxpool then double conv"""

    def __init__(self, out_channels):
        super().__init__()
        self.maxpool_conv = tf.keras.Sequential([
            layers.MaxPool2D(),
            DoubleConv(out_channels)
        ])

    def call(self, x, training=False, exp_stat=False):
        if exp_stat:
            x = self.maxpool_conv.layers[0](x)
            return self.maxpool_conv.layers[1](x, training=training, exp_stat=exp_stat)
        return self.maxpool_conv(x, training=training)


class Up(layers.Layer):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
            self.conv = DoubleConv(out_channels, in_channels // 2)
        else:
            self.up = layers.Conv2DTranspose(in_channels // 2, kernel_size=2, padding='same', strides=2)
            self.conv = DoubleConv(out_channels)

    def call(self, inputs, training=False, exp_stat=False):
        x1, x2 = inputs
        x1 = self.up(x1)
        x = layers.concatenate([x2, x1], axis=-1)
        if exp_stat:
            return self.conv(x, training=training, exp_stat=exp_stat)
        return self.conv(x, training=training)


class OutConv(layers.Layer):
    def __init__(self, out_channels):
        super(OutConv, self).__init__()
        self.conv = layers.Conv2D(out_channels, kernel_size=1, padding='same')

    def call(self, inputs, *args, **kwargs):
        return self.conv(inputs)


class UNet(tf.keras.Model):
    def __init__(self, n_classes, bilinear=True):
        super().__init__()
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(64)
        self.down1 = Down(128)
        self.down2 = Down(256)
        self.down3 = Down(512)
        factor = 2 if bilinear else 1
        self.down4 = Down(1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(n_classes)

    def call(self, x, training=False, exp_stat=False, *args, **kwargs):
        if exp_stat:
            pre_bn_feats = []

            x1, f1 = self.inc(x, training=training, exp_stat=True)
            pre_bn_feats += f1

            x2, f2 = self.down1(x1, training=training, exp_stat=True)
            pre_bn_feats += f2

            x3, f3 = self.down2(x2, training=training, exp_stat=True)
            pre_bn_feats += f3

            x4, f4 = self.down3(x3, training=training, exp_stat=True)
            pre_bn_feats += f4

            x5, f5 = self.down4(x4, training=training, exp_stat=True)
            pre_bn_feats += f5

            x, f = self.up1((x5, x4), training=training, exp_stat=True)
            pre_bn_feats += f

            x, f = self.up2((x, x3), training=training, exp_stat=True)
            pre_bn_feats += f

            x, f = self.up3((x, x2), training=training, exp_stat=True)
            pre_bn_feats += f

            x, f = self.up4((x, x1), training=training, exp_stat=True)
            pre_bn_feats += f
            logits = self.outc(x)
            return logits, pre_bn_feats
        else:
            x1 = self.inc(x, training=training)
            x2 = self.down1(x1, training=training)
            x3 = self.down2(x2, training=training)
            x4 = self.down3(x3, training=training)
            x5 = self.down4(x4, training=training)
            x = self.up1((x5, x4), training=training)
            x = self.up2((x, x3), training=training)
            x = self.up3((x, x2), training=training)
            x = self.up4((x, x1), training=training)
            logits = self.outc(x)
        return logits
