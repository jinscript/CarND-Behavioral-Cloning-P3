from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import csv
import cv2
import numpy as np

# simulator generated data
DATA_DIR = './data'


class DataProcessor(object):
    """
    Data processor
    """

    WIDTH = 320
    HEIGHT = 75
    NUM_CHANNELS = 1
    BATCH_SIZE = 128

    @classmethod
    def load(cls):
        """
        Load data from csv and create generators
        :return: train and validate generators and length of train set and validate set
        """
        path = os.path.join(DATA_DIR, 'driving_log.csv')
        samples = cls._extract_samples_from_csv(path)
        shuffle(samples)

        train_samples, valid_samples = train_test_split(samples, test_size=0.2)

        return cls._generator(train_samples, cls.BATCH_SIZE), \
               cls._generator(valid_samples, cls.BATCH_SIZE), \
               len(train_samples), len(valid_samples)

    @classmethod
    def process(cls, image):
        """
        Pre-process an image. Techniques used:
        - Cropping: only keep image segment with road information to reduce noise
        - Grayscale
        This methods is used in drive.py as well
        :param image: image
        :return:      processed image
        """
        image = image[65:140,:,:]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = np.reshape(image, (cls.HEIGHT, cls.WIDTH, cls.NUM_CHANNELS))
        return image / 255 - 0.5

    @classmethod
    def _generator(cls, samples, batch_size):
        """
        Create generators for data set

        :param samples:    list of image
        :param batch_size: batch size
        :return:           a generator
        """

        n_samples = len(samples)

        while True:
            for i in range(0, n_samples, batch_size):
                batch_samples = samples[i: i + batch_size]

                images, targets = [], []
                for sample in batch_samples:
                    image_path, angle = sample
                    image = cv2.imread(image_path)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(cls.process(image))
                    targets.append(angle)

                yield np.array(images), np.array(targets)

    @classmethod
    def _extract_samples_from_csv(cls, path):
        """
        Extract image path and steer angle from csv
        Add +0.2 for left image
        Add -0.2 for right image
        :param path: data dir
        :return:     a list of (image_path, steer_angle)
        """

        lines = []
        with open(path) as f:
            reader = csv.reader(f)
            for line in reader:

                center, left, right, angle, throttle, brake, speed = line

                dirs = []
                dirs.extend(path.split('/')[:-1])
                dirs.extend(center.split('/')[-2:])
                center = os.path.join(*dirs)

                dirs = []
                dirs.extend(path.split('/')[:-1])
                dirs.extend(left.split('/')[-2:])
                left = os.path.join(*dirs)

                dirs = []
                dirs.extend(path.split('/')[:-1])
                dirs.extend(right.split('/')[-2:])
                right = os.path.join(*dirs)

                if os.path.exists(center):
                    lines.append((center, float(angle)))
                  
                if os.path.exists(left):
                    lines.append((left, float(angle) + 0.2))
                    
                if os.path.exists(right):
                    lines.append((right, float(angle) - 0.2))

        return lines
