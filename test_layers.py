from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import layers
import hparams as hp

_BATCH_SIZE = 32
_TOTAL_SEQUENCE_LENGTH = 20
_INPUT_DEPTH = 256
_NUM_CELLS = 6
_CELL_NUMBER = 3

_RESIZE_EXEMPT_LAYER_PREFIXES = [
    "depthwise_conv", "squeeze_and_excitation", "identity", "lightweight_conv",
]


def _apply_encoder_layer(translation_layer, output_depth, nonpadding_list):
    """Applies an encoder layer with basic arguments."""

    input_tensor = tf.random.uniform(
        [_BATCH_SIZE, _TOTAL_SEQUENCE_LENGTH, _INPUT_DEPTH]) / 4.0
    nonpadding = tf.constant(nonpadding_list)
    residual_tensor = tf.random.uniform(
        [_BATCH_SIZE, _TOTAL_SEQUENCE_LENGTH, output_depth])
    hparams = hp.transformer_base()

    return translation_layer.apply_layer(
        input_tensor,
        residual_tensor,
        output_depth,
        tf.nn.relu,
        hparams,
        "",
        mask_future=False,
        nonpadding=nonpadding,
        layer_preprocess_fn=None,
        postprocess_dropout=True)


def _create_nonpadding_list():
    """Creates the `nonpadding_list` for applying the encoder layers."""
    nonpadding_list = []
    for i in range(_BATCH_SIZE):
        nonpadding_list.append([1.0] * min(i + 2, _TOTAL_SEQUENCE_LENGTH) +
                               [0.0] * max((_TOTAL_SEQUENCE_LENGTH - i - 2), 0))
    return nonpadding_list


def _get_empirical_parameters(layer):
    """Gets the number of parameters built into the current Tensorflow graph."""
    trainable_variables_list = layer.trainable_variables

    empirical_num_params = 0
    for variable_tensor in trainable_variables_list:
        empirical_num_params += np.prod(variable_tensor.shape)

    return empirical_num_params


class LayersTest(parameterized.TestCase, tf.test.TestCase):
    """Tests params, residual capabilities, padding leaks, and output shape."""

    # Test that the encoder registry contains all the expected layers.
    def test_encoder_registry(self):
        encoder_layers = [
            # "separable_conv_3x1",
            # "separable_conv_5x1",
            # "separable_conv_7x1",
            # "separable_conv_9x1",
            # "separable_conv_11x1",
            # "separable_conv_13x1",
            # "separable_conv_15x1",
            # "standard_conv_1x1",
            # "standard_conv_3x1",
            # "standard_conv_5x1",
            # "depthwise_conv_3x1",
            # "depthwise_conv_5x1",
            # "depthwise_conv_7x1",
            # "dilated_conv_3x1",
            # "dilated_conv_5x1",
            "standard_attention",
            # "identity",
            "attention_4_heads",
            "attention_16_heads",
            "attention_32_heads",
            # "gated_linear_unit",
            # "lightweight_conv_3x1_r_1",
            # "lightweight_conv_3x1_r_4",
            # "lightweight_conv_3x1_r_16",
            # "lightweight_conv_5x1_r_1",
            # "lightweight_conv_5x1_r_4",
            # "lightweight_conv_5x1_r_16",
            # "lightweight_conv_7x1_r_1",
            # "lightweight_conv_7x1_r_4",
            # "lightweight_conv_7x1_r_16",
            # "lightweight_conv_15x1_r_1",
            # "lightweight_conv_15x1_r_4",
            # "lightweight_conv_15x1_r_16",
        ]
        self.assertSameElements(encoder_layers,
                                layers.ENCODER_LAYERS.get_layer_names())

    # Test that the decoder registry contains all the expected layers.

    # Test encoder layer. This includes checking that output dims are as
    # expected, checking that num_params() agrees with the empirical number of
    # variables produced, and that information does not leak from 0 padded
    # areas of the input.
    @parameterized.parameters(
        itertools.product(layers.ENCODER_LAYERS.get_layer_names(),
                          (256, 128, 512)))  # output_depth :128, 256, 512
    def test_encoder_layer(self, translation_layer_name, output_depth):
        # with self.test_session(graph=tf.Graph()) as sess:
        nonpadding_list = _create_nonpadding_list()
        for prefix in _RESIZE_EXEMPT_LAYER_PREFIXES:
            if prefix in translation_layer_name:
                output_depth = _INPUT_DEPTH
        translation_layer = layers.ENCODER_LAYERS.get(translation_layer_name)
        output_tensor = _apply_encoder_layer(translation_layer, output_depth,
                                             nonpadding_list)

        # Check that the output shape is as expected.
        self.assertEqual(output_tensor.shape.as_list(),
                         [_BATCH_SIZE, _TOTAL_SEQUENCE_LENGTH, output_depth])

        # Check that the number of parameters is as expected.
        # empirical_num_params = _get_empirical_parameters(translation_layer)
        # print("---------------", empirical_num_params)
        # reported_num_params = translation_layer.num_params(
        #     _INPUT_DEPTH, output_depth)
        # self.assertEqual(empirical_num_params, reported_num_params)

        # Make sure padding is applied properly (no leaks).
        # sess.run(tf.global_variables_initializer())
        # output = sess.run(output_tensor)

        output = output_tensor

        for i, j in itertools.product(
                range(_BATCH_SIZE), range(_TOTAL_SEQUENCE_LENGTH)):
            if nonpadding_list[i][j] == 0:
                self.assertAllEqual(output[i][j], np.array([0] * output_depth),
                                    "Output row %s, column %s not zeroed out." % (i, j))



# Test decoder layer. This includes checking that output dims are as
# expected, checking that num_params() agrees with the empirical number of
# variables produced, and that temporal information does not leak.
# @parameterized.parameters(
#     itertools.product(layers.DECODER_LAYERS.get_layer_names(),
#                       (256, 128, 512)))
# def test_decoder_layer(self, translation_layer_name, output_depth):
#   with self.test_session(graph=tf.Graph()) as sess:
#
#     # Check that the output shape is as expected.
#     input_tensor = tf.placeholder(
#         tf.float32, [_BATCH_SIZE, _TOTAL_SEQUENCE_LENGTH, _INPUT_DEPTH])
#     encoder_depth = int(_INPUT_DEPTH / 2)
#     for prefix in _RESIZE_EXEMPT_LAYER_PREFIXES:
#       if prefix in translation_layer_name:
#         output_depth = _INPUT_DEPTH
#     translation_layer = layers.DECODER_LAYERS.get(translation_layer_name)
#     output_tensor = _apply_decoder_layer(translation_layer, input_tensor,
#                                          output_depth, encoder_depth)
#     self.assertEqual(output_tensor.shape.as_list(),
#                      [_BATCH_SIZE, _TOTAL_SEQUENCE_LENGTH, output_depth])
#
#     # Check that the number of parameters is as expected.
#     empirical_num_params = _get_empirical_parameters()
#     reported_num_params = translation_layer.num_params(
#         _INPUT_DEPTH,
#         output_depth,
#         encoder_depth=encoder_depth)
#     self.assertEqual(empirical_num_params, reported_num_params)
#
#     # Check that there is no temporal information leak. Specifically, check
#     # that values before `test_index` remain unchanged, while the values
#     # after it have changed. Sums are used because two values could
#     # potentially be the same between the zero and non-zero portions, even
#     # if the masking is working correctly. Note: This assumes that the
#     # output at t is dependent on the input at t.
#     feed_input = np.random.random(
#         [_BATCH_SIZE, _TOTAL_SEQUENCE_LENGTH, _INPUT_DEPTH]) / 10.0
#     test_index = int(_TOTAL_SEQUENCE_LENGTH / 2)
#     transformed_feed_input = _zero_after_index_copy(feed_input, test_index)
#
#     # Produce the outputs for both types of input.
#     feed_dict = {
#         v: np.random.rand(*v.shape.as_list()) - .5
#         for v in tf.all_variables()
#     }
#     feed_dict[input_tensor] = feed_input
#     control_output = sess.run(output_tensor, feed_dict)
#
#     feed_dict[input_tensor] = transformed_feed_input
#     variable_output = sess.run(output_tensor, feed_dict)
#
#     self.assertAllClose(
#         control_output[:, :test_index + 1],
#         variable_output[:, :test_index + 1],
#         rtol=1)
#
#     with self.assertRaises(
#         AssertionError,
#         msg="Time-masked portion of output too close to control output."):
#       self.assertAllClose(
#           control_output[:, test_index + 1:],
#           variable_output[:, test_index + 1:],
#           rtol=1)

if __name__ == "__main__":
    tf.test.main()
