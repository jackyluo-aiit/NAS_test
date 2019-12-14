from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import layers

_BATCH_SIZE = 32
_TOTAL_SEQUENCE_LENGTH = 20
_INPUT_DEPTH = 256
_NUM_CELLS = 6
_CELL_NUMBER = 3

input_tensor = tf.random.uniform(
      [_BATCH_SIZE, _TOTAL_SEQUENCE_LENGTH, _INPUT_DEPTH]) / 4.0
print(input_tensor)
