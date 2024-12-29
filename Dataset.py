import tensorflow as tf
from tensorflow import keras as kr
import HyperParameters as hp
import tensorflow_datasets as tfds


if hp.is_cifar:
    @tf.function
    def _normalize(data):
        imgs = tf.cast(data['image'], 'float32') / 127.5 - 1.0
        return imgs

    def load_datasets():
        dataset = tfds.load('cifar10')
        train_dataset = dataset['train'].shuffle(1000).batch(hp.batch_size, drop_remainder=True).map(_normalize)
        test_dataset = dataset['test'].shuffle(1000).batch(hp.batch_size, drop_remainder=True).map(_normalize)

        return train_dataset, test_dataset



else:
    @tf.function
    def _normalize(data):
        imgs = tf.cast(data['image'], 'float32') / 127.5 - 1.0
        return imgs

    def load_datasets():
        dataset = tfds.load('stl10')
        train_dataset = dataset['unlabelled'].shuffle(1000).batch(hp.batch_size, drop_remainder=True).map(_normalize)
        test_dataset = dataset['unlabelled'].take(20000).shuffle(1000).batch(hp.batch_size, drop_remainder=True).map(_normalize)
        return train_dataset, test_dataset
