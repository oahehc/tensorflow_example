import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime
import matplotlib.pyplot as plt


# # DATASET
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) # read mnist data