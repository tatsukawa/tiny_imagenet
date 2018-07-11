import os
import numpy as np
import tensorflow as tf
import random
import glob

class Loader:

    def __init__(self):
        self.ROOT_PATH = '../data/tiny-imagenet-200'

        self.raw_labels = []
        with open(os.path.join(self.ROOT_PATH, 'wnids.txt'), 'r') as f:
            for line in f:
                line = line.strip()
                self.raw_labels.append(line)

        self.NUM_CLASSES = len(self.raw_labels)
        self.convert_label_to_num = {label:num for label, num in zip(self.raw_labels, range(len(self.raw_labels)))}
        self.convert_num_to_label = {num:label for label, num in zip(self.raw_labels, range(len(self.raw_labels)))}
#        self.raw_label_description = np.genfromtxt(os.path.join(self.ROOT_PATH, 'words.txt'), dtype=str)

        tf.logging.info('NUM_LABELS: {0}'.format(len(self.raw_labels)))
        tf.logging.info('convert_labels_to_num example:  {0}'.format(self.convert_label_to_num[self.raw_labels[0]]))

        self.W = 64
        self.H = 64
        self.CHANNELS = 3


    def _get_test_data(self):
        path = os.path.join('../data/test')

        images = []
        names = []

        for path in glob.glob('{}/*.jpeg'.format(path)):
            images.append(path)
            names.append(os.path.basename(path))

        return images, names


    def get_test_data(self):
        """

        :param num_threads:
        :return:
        """
        tf.logging.info('Get test data')
        X, names = self._get_test_data()
        dummy_labels = [0 for i in range(len(X))]
        X_tensor = tf.constant(X)
        Y_tensor = tf.constant(dummy_labels)

        data = tf.data.Dataset.from_tensor_slices(
            (X_tensor, Y_tensor)
        )

        data = data.map(self._parse_data)
        data = data.map(self._resize_data)
        data = data.map(self._normalize_data)
        data = data.batch(int(len(X)))
        data = data.repeat(1)

        iterator = tf.data.Iterator.from_structure(
            data.output_types, data.output_shapes
        )

        next_element = iterator.get_next()
        init_op = iterator.make_initializer(data)

        return next_element, init_op, names


    def _get_training_image_path_and_labels(self):
        path = os.path.join(self.ROOT_PATH, 'train')

        images = []
        labels = []

        MAX_DATA_NUM = 500

        for label in self.raw_labels:
            p = os.path.join(path, label, 'images')

            for i in range(MAX_DATA_NUM):
                _p = os.path.join(p, '{0}_{1}.JPEG'.format(label, i))
                images.append(_p)
                labels.append(self.convert_label_to_num[label])

        tf.logging.info('train data example: {0}, {1}'.format(images[0], labels[0]))

        return images, labels


    def _parse_data(self, image_path, label):
        image = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        label = tf.one_hot(label, depth=len(self.raw_labels))
        return image, label


    def _get_validation_image_and_labels(self):
        path = os.path.join(self.ROOT_PATH, 'val')

        images = []
        labels = []

        with open(os.path.join(path, 'val_annotations.txt')) as f:
            for line in f:
                var = line.split('\t')
                image_path = os.path.join(path, 'images', var[0])
                label = self.convert_label_to_num[var[1]]
                images.append(image_path)
                labels.append(label)

        tf.logging.info('validation data example: {0}, {1}'.format(images[0], labels[0]))

        return images, labels


    def get_validation_data(self, num_threads=4):
        """we need to get validation data at only one time

        :param num_threads:
        :return:
        """
        tf.logging.info('Get validation data')

        X, Y = self._get_validation_image_and_labels()

        num_prefetch = 64
        X_tensor = tf.constant(X)
        Y_tensor = tf.constant(Y)


        data = tf.data.Dataset.from_tensor_slices(
            (X_tensor, Y_tensor)
        )

        data = data.map(self._parse_data,       num_parallel_calls=num_threads).prefetch(num_prefetch)
        data = data.map(self._resize_data,      num_parallel_calls=num_threads).prefetch(num_prefetch)
        data = data.map(self._normalize_data,   num_parallel_calls=num_threads).prefetch(num_prefetch)
        data = data.batch(int(len(X)/20))
        data = data.repeat(10000)

        iterator = tf.data.Iterator.from_structure(
            data.output_types, data.output_shapes
        )

        next_element = iterator.get_next()

        init_op = iterator.make_initializer(data)

        return next_element, init_op


    def get_train_batch(self, is_augment=False, batch_size=128, num_threads=4):
        """ref: https://github.com/HasnainRaz/Tensorflow-input-pipeline/blob/master/utility.py

        :param is_augment:
        :param batch_size:
        :param num_threads:
        :return:
        """
        tf.logging.info('Get train batch data')

        X, Y = self._get_training_image_path_and_labels()

        num_prefetch = 64
        num_shuffle = len(X)

        X_tensor = tf.constant(X)
        Y_tensor = tf.constant(Y)

        data = tf.data.Dataset.from_tensor_slices(
            (X_tensor, Y_tensor)
        )

        data = data.map(self._parse_data, num_parallel_calls=num_threads).prefetch(num_prefetch)

        if is_augment:
            data = data.map(self._corrupt_brightness,    num_parallel_calls=num_threads).prefetch(num_prefetch)
            data = data.map(self._corrupt_contrast,      num_parallel_calls=num_threads).prefetch(num_prefetch)
            data = data.map(self._corrupt_saturation,    num_parallel_calls=num_threads).prefetch(num_prefetch)
            data = data.map(self. _crop_random,          num_parallel_calls=num_threads).prefetch(num_prefetch)
            data = data.map(self._flip_left_right,       num_parallel_calls=num_threads).prefetch(num_prefetch)

        data = data.map(self._resize_data,      num_parallel_calls=num_threads).prefetch(num_prefetch)
        data = data.map(self._normalize_data,   num_parallel_calls=num_threads).prefetch(num_prefetch)
        data = data.shuffle(num_shuffle)
        data = data.batch(batch_size)
        data = data.repeat(10000)

        iterator = tf.data.Iterator.from_structure(
            data.output_types, data.output_shapes
        )

        next_element = iterator.get_next()

        init_op = iterator.make_initializer(data)

        return next_element, init_op


    def _corrupt_brightness(self, image, label):
        """Radnomly applies a random brightness change."""
        cond_brightness = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_brightness, lambda: tf.image.random_hue(image, 0.1), lambda: tf.identity(image))
        return image, label


    def _corrupt_contrast(self, image, label):
        """Randomly applies a random contrast change."""
        cond_contrast = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(image, 0.2, 1.8), lambda: tf.identity(image))
        return image, label


    def _corrupt_saturation(self, image, label):
        """Randomly applies a random saturation change."""
        cond_saturation = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_saturation, lambda: tf.image.random_saturation(image, 0.2, 1.8), lambda: tf.identity(image))
        return image, label


    def _crop_random(self, image, label):
        """Randomly crops image in accord."""
        seed = random.random()
        cond_crop_image = tf.cast(tf.random_uniform([], maxval=2, dtype=tf.int32, seed=seed), tf.bool)

        image = tf.cond(cond_crop_image, lambda: tf.random_crop(
            image, [int(self.H * 0.85), int(self.W * 0.85), 3], seed=seed), lambda: tf.identity(image))

        image = tf.expand_dims(image, axis=0)
        image = tf.image.resize_images(image, [self.H, self.W])
        image = tf.squeeze(image, axis=0)

        return image, label


    def _flip_left_right(self, image, label):
        """Randomly flips image left or right in accord."""
        seed = random.random()
        image = tf.image.random_flip_left_right(image, seed=seed)

        return image, label


    def _normalize_data(self, image, label):
        """Normalize image within range 0-1."""
        image = tf.cast(image, tf.float32)
        image = image / 255.0

        return image, label


    def _resize_data(self, image, label):
        """Resizes images to smaller dimensions."""
        # TODO: implement but not need to do
        return image, label

