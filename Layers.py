import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp


class Blur2D(kr.layers.Layer):
    def __init__(self, upscale=False, downscale=False):
        super().__init__()
        self.upscale = upscale
        self.downscale = downscale
        assert (upscale and downscale) != True

    def build(self, __input_shape):
        _, self.in_c, self.in_h, self.in_w = __input_shape
        kernel = tf.cast([1, 3, 3, 1], 'float32')
        kernel = tf.tensordot(kernel, kernel, axes=0)
        kernel = kernel / tf.reduce_sum(kernel)
        self.kernel = kernel[:, :, tf.newaxis, tf.newaxis]

    def call(self, __inputs):
        if self.upscale:
            ftr_maps = tf.reshape(__inputs, [-1, 1, self.in_h, 1, self.in_w, 1])
            ftr_maps = tf.pad(ftr_maps, [[0, 0], [0, 0], [0, 0], [0, 1], [0, 0], [0, 1]])
            ftr_maps = tf.reshape(ftr_maps, [-1, 1, self.in_h * 2, self.in_w * 2])
            ftr_maps = tf.nn.conv2d(ftr_maps, self.kernel * 4, strides=1, padding='SAME', data_format='NCHW')
            return tf.reshape(ftr_maps, [-1, self.in_c, self.in_h * 2, self.in_w * 2])
        elif self.downscale:
            ftr_maps = tf.reshape(__inputs, [-1, 1, self.in_h, self.in_w])
            ftr_maps = tf.nn.conv2d(ftr_maps, self.kernel, strides=2, padding='SAME', data_format='NCHW')
            return tf.reshape(ftr_maps, [-1, self.in_c, self.in_h // 2, self.w // 2])
        else:
            ftr_maps = tf.reshape(__inputs, [-1, 1, self.in_h, self.in_w])
            ftr_maps = tf.nn.conv2d(ftr_maps, self.kernel, strides=1, padding='SAME', data_format='NCHW')
            return tf.reshape(ftr_maps, [-1, self.in_c, self.in_h, self.in_w])


#----------------------------------------------------------------------------

class Dense(kr.layers.Layer):
    def __init__(self, units, activation=kr.activations.linear, use_bias=True, lr_scale=1.0):
        super().__init__()
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        self.lr_scale = lr_scale

    def build(self, __input_shape):
        in_d = __input_shape[-1]
        self.multiplier = tf.sqrt(1 / tf.cast(in_d, 'float32'))
        self.kernel = tf.Variable(tf.random.normal([in_d, self.units]) / self.lr_scale, name=self.name + '_kernel')
        if self.use_bias:
            self.bias = tf.Variable(tf.zeros([1, self.units]), name=self.name + '_bias')

    def call(self, __inputs):
        ftr_vec = __inputs @ self.kernel * self.multiplier * self.lr_scale
        if self.use_bias:
            ftr_vec += self.bias * self.lr_scale
        return self.activation(ftr_vec)

#----------------------------------------------------------------------------

class Conv2D(kr.layers.Layer):
    def __init__(self, filter_size, kernel_size, activation=kr.activations.linear, use_bias=True, upscale=False, downscale=False):
        super().__init__()
        self.filter_size = filter_size
        self.kernel_size = kernel_size
        self.activation = activation
        self.use_bias = use_bias
        self.upscale = upscale
        self.downscale = downscale
        assert (upscale and downscale) != True

    def build(self, __input_shape):
        _, self.in_c, self.in_h, self.in_w = __input_shape

        if self.upscale:
            self.blur = Blur2D()
            self.kernel = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, self.filter_size, self.in_c]),
                                      name=self.name + '_kernel')
            self.multiplier = tf.sqrt(1 / tf.cast(self.kernel_size * self.kernel_size * self.filter_size, 'float32'))
        elif self.downscale:
            self.blur = Blur2D()
            self.kernel = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, self.in_c, self.filter_size]),
                                      name=self.name + '_kernel')
            self.multiplier = tf.sqrt(1 / tf.cast(self.kernel_size * self.kernel_size * self.in_c, 'float32'))
        else:
            self.kernel = tf.Variable(tf.random.normal([self.kernel_size, self.kernel_size, self.in_c, self.filter_size]),
                                      name=self.name + '_kernel')
            self.multiplier = tf.sqrt(1 / tf.cast(self.kernel_size * self.kernel_size * self.in_c, 'float32'))

        if self.use_bias:
            self.bias = tf.Variable(tf.zeros([1, self.filter_size, 1, 1]), name=self.name + '_bias')

    def call(self, __inputs):
        if self.upscale:
            ftr_maps = self.blur(tf.nn.conv2d_transpose(__inputs, self.kernel * self.multiplier * 4,
                                                    output_shape=[tf.shape(__inputs)[0], self.filter_size, self.in_h * 2, self.in_w * 2],
                                                    strides=2, padding='SAME', data_format='NCHW'))
        elif self.downscale:
            ftr_maps = tf.nn.conv2d(self.blur(__inputs), self.kernel * self.multiplier,
                                    strides=2, padding='SAME', data_format='NCHW')
        else:
            ftr_maps = tf.nn.conv2d(__inputs, self.kernel * self.multiplier,
                                    strides=1, padding='SAME', data_format='NCHW')

        if self.use_bias:
            ftr_maps += self.bias

        return self.activation(ftr_maps)


class Book(kr.layers.Layer):
    def __init__(self, page_shape):
        super().__init__()
        self.page_shape = page_shape

    def build(self, __input_shape):
        self.book = tf.Variable(tf.random.normal([1, hp.lbl_dim, hp.ctg_dim, self.page_shape[0] * self.page_shape[1] * self.page_shape[2]]), name=self.name + '_book')
        self.reshape_layer = kr.layers.Reshape([self.page_shape[0] * hp.lbl_dim, self.page_shape[1], self.page_shape[2]])
    def call(self, __inputs):
        return self.reshape_layer(tf.reduce_sum(self.book * __inputs[:, :, :, tf.newaxis], axis=2))


if hp.is_cifar:
    filter_sizes = [256, 512, 512]
else:
    filter_sizes = [128, 256, 512, 512]
act = tf.nn.leaky_relu

class Generator(kr.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, __input_shape):
        cnt_vec = kr.Input([hp.cnt_dim])
        cnd_vec = kr.Input([hp.lbl_dim, hp.ctg_dim])
        if hp.use_codebook:
            if hp.is_cifar:
                ftr_vec = Dense(units=512 * 4 * 4, activation=act)(cnt_vec)
                ftr_maps = kr.layers.Reshape([512, 4, 4])(ftr_vec)
                ftr_maps = tf.concat([ftr_maps, Book([512 // hp.lbl_dim, 4, 4])(cnd_vec)], axis=1)
            else:
                ftr_vec = Dense(units=512 * 6 * 6, activation=act)(cnt_vec)
                ftr_maps = kr.layers.Reshape([512, 6, 6])(ftr_vec)
                ftr_maps = tf.concat([ftr_maps, Book([512 // hp.lbl_dim, 6, 6])(cnd_vec)], axis=1)

        else:
            ctg_dim = tf.cast(hp.ctg_dim, 'float32')
            norm_cnd_vec = (cnd_vec - 1 / ctg_dim) * ctg_dim / tf.sqrt(ctg_dim - 1)
            ftr_vec = tf.concat([cnt_vec, kr.layers.Flatten()(norm_cnd_vec)], axis=-1)
            if hp.is_cifar:
                ftr_vec = Dense(units=1024 * 4 * 4, activation=act)(ftr_vec)
                ftr_maps = kr.layers.Reshape([1024, 4, 4])(ftr_vec)
            else:
                ftr_vec = Dense(units=1024 * 6 * 6, activation=act)(ftr_vec)
                ftr_maps = kr.layers.Reshape([1024, 6, 6])(ftr_vec)

        for filter_size in reversed(filter_sizes):
            skp_maps = Conv2D(filter_size=filter_size, kernel_size=1, use_bias=False, upscale=True)(ftr_maps)
            ftr_maps = Conv2D(filter_size=filter_size, kernel_size=3, activation=act, upscale=True)(ftr_maps)
            ftr_maps = Conv2D(filter_size=filter_size, kernel_size=3, activation=act)(ftr_maps)
            ftr_maps = (ftr_maps + skp_maps) / tf.sqrt(2.0)

        fake_img = Conv2D(filter_size=hp.img_chn, kernel_size=1)(ftr_maps)
        fake_img = tf.transpose(fake_img, [0, 2, 3, 1])

        self.model = kr.Model([cnd_vec, cnt_vec], fake_img)

    def call(self, __inputs):
        return self.model(__inputs)


class Encoder(kr.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, __input_shape):
        ftr_maps = inp_img = kr.Input([hp.img_res, hp.img_res, hp.img_chn])
        ftr_maps = tf.transpose(ftr_maps, [0, 3, 1, 2])
        ftr_maps = Conv2D(filter_size=filter_sizes[0], kernel_size=1, activation=act)(ftr_maps)

        for filter_size in filter_sizes:
            skip_maps = Conv2D(filter_size=filter_size, kernel_size=1, use_bias=False, downscale=True)(ftr_maps)
            ftr_maps = Conv2D(filter_size=filter_size, kernel_size=3, activation=act)(ftr_maps)
            ftr_maps = Conv2D(filter_size=filter_size, kernel_size=3, activation=act, downscale=True)(ftr_maps)
            ftr_maps = (skip_maps + ftr_maps) / tf.sqrt(2.0)
        ftr_maps = Conv2D(filter_size=1024, kernel_size=3, activation=act)(ftr_maps)
        ftr_vec = kr.layers.Flatten()(ftr_maps)
        ctg_vec = kr.layers.Reshape([hp.lbl_dim, hp.ctg_dim])(Dense(units=hp.lbl_dim * hp.ctg_dim)(ftr_vec))
        self.model = kr.Model(inp_img, ctg_vec)

    def call(self, __inputs):
        return self.model(__inputs)
