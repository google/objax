# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unittests for Convolution Layer."""

import unittest

import jax.numpy as jn

import objax


class TestConv(unittest.TestCase):
    def test_on_conv2d_three_by_three(self):
        """
        Pass an input through a three-by-three convolution filter and
        test the shape and contents of the output.
        """

        # Channels/Colors, #filters, filter_size (square)
        conv_filter = objax.nn.Conv2D(1, 1, 3, padding=objax.ConvPadding.VALID)
        weights = objax.TrainVar(jn.array([[[[1., 2., 1.], [1., 2., 1.], [1., 1., 1.]]]]).transpose((2, 3, 1, 0)))
        conv_filter.w = weights
        image = jn.array([[[[2., 1., 3., 4.],
                            [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        # NCHW: Batch, Channels/Colors, Height, Width
        features = conv_filter(image)
        expected_features = jn.array([[[[61.0, 72.], [106., 117.]]]])
        self.assertEqual(features.shape, (1, 1, 2, 2))
        self.assertTrue(jn.array_equal(features, expected_features))

    def test_on_conv2d_two_by_two(self):
        """
        Pass an input through a two-by-two convolution filter and
        test the shape and contents of the output.
        """

        # Channels/Colors, #filters, filter_size (square)
        conv_filter = objax.nn.Conv2D(1, 1, 2, padding=objax.ConvPadding.VALID)
        weights = objax.TrainVar(jn.array([[[[1., 2.], [3., 4.]]]]).transpose((2, 3, 1, 0)))
        conv_filter.w = weights
        image = jn.array([[[[2., 1., 3., 4.],
                            [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        # NCHW: Batch, Channels/Colors, Height, Width
        features = conv_filter(image)
        expected_features = jn.array([[[[43., 53., 64.], [84., 94., 104.], [124., 134., 144.]]]])
        self.assertEqual(features.shape, (1, 1, 3, 3))
        self.assertTrue(jn.array_equal(features, expected_features))

    def test_on_conv2d_padding(self):
        """
        Pass an input through a two-by-two convolution filter with padding and
        test the shape and contents of the output.
        """

        # Channels/Colors, #filters, filter_size (square)
        conv_filter = objax.nn.Conv2D(1, 1, 2, padding=objax.ConvPadding.SAME)
        weights = objax.TrainVar(jn.array([[[[1., 2.], [3., 4.]]]]).transpose((2, 3, 1, 0)))
        conv_filter.w = weights
        image = jn.array([[[[2., 1., 3., 4.],
                            [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        # NCHW: Batch, Channels/Colors, Height, Width
        features = conv_filter(image)
        expected_features = jn.array([[[[43., 53., 64., 28.], [84., 94., 104., 44.],
                                        [124., 134., 144., 60.], [41., 44., 47., 16.]]]])
        self.assertEqual(features.shape, (1, 1, 4, 4))
        self.assertTrue(jn.array_equal(features, expected_features))

    def test_on_conv2d_stride(self):
        """
        Pass an input through a two-by-two convolution filter with stride=2
        and test the shape and contents of the output.
        """

        # Channels/Colors, #filters, filter_size (square)
        conv_filter = objax.nn.Conv2D(1, 1, 2, strides=2, padding=objax.ConvPadding.VALID)
        weights = objax.TrainVar(jn.array([[[[1., 2.], [3., 4.]]]]).transpose((2, 3, 1, 0)))
        conv_filter.w = weights
        image = jn.array([[[[2., 1., 3., 4.],
                            [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        # NCHW: Batch, Channels/Colors, Height, Width
        features = conv_filter(image)
        expected_features = jn.array([[[[43., 64.], [124., 144.]]]])
        self.assertEqual(features.shape, (1, 1, 2, 2))
        self.assertTrue(jn.array_equal(features, expected_features))

    def test_on_conv2d_two_filters(self):
        """
        Pass an input through two two-by-two convolution filter
        and test the shape and contents of the output.
        """

        # Channels/Colors, #filters, filter_size (square)
        conv_filter = objax.nn.Conv2D(1, 2, 2, padding=objax.ConvPadding.VALID)
        weights = objax.TrainVar(jn.array([[[[1., 2.], [3., 4.]]], [[[1., 2.], [3., 4.]]]]).transpose((2, 3, 1, 0)))
        conv_filter.w = weights
        image = jn.array([[[[2., 1., 3., 4.], [5., 6., 7., 8.],
                            [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        # NCHW: Batch, Channels/Colors, Height, Width
        features = conv_filter(image)
        expected_features = jn.array([[[[43., 53., 64.], [84., 94., 104.], [124., 134., 144.]],
                                       [[43., 53., 64.], [84., 94., 104.], [124., 134., 144.]]]])
        self.assertEqual(features.shape, (1, 2, 3, 3))
        self.assertTrue(jn.array_equal(features, expected_features))

    def test_on_conv2d_two_channels(self):
        """
        Pass an input through two-by-two convolution filter
        with two channels and test the shape and contents of the output.
        """

        # Channels/Colors, #filters, filter_size (square)
        conv_filter = objax.nn.Conv2D(2, 1, 2, padding=objax.ConvPadding.VALID)
        weights = objax.TrainVar(jn.array([[[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]]).transpose((2, 3, 1, 0)))
        conv_filter.w = weights
        image = jn.array([[[[2., 1., 3., 4.], [5., 6., 7., 8.],
                            [9., 10., 11., 12.], [13., 14., 15., 16.]],
                           [[2., 1., 3., 4.], [5., 6., 7., 8.],
                            [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        # NCHW: Batch, Channels/Colors, Height, Width
        features = conv_filter(image)
        expected_features = jn.array([[[[86., 106., 128.], [168., 188., 208.],
                                        [248., 268., 288.]]]])
        self.assertEqual(features.shape, (1, 1, 3, 3))
        self.assertTrue(jn.array_equal(features, expected_features))

    def test_on_conv2d_dilation_padding_same(self):
        """
        Pass an input through two-by-two convolution filter
        with one channel and test the shape and contents of the output.
        """

        # Channels/Colors, #filters, filter_size (square)
        conv_filter = objax.nn.Conv2D(1, 1, 2, dilations=2, padding=objax.ConvPadding.SAME)
        weights = objax.TrainVar(jn.array([[[[1., 2.], [3., 4.]]]]).transpose((2, 3, 1, 0)))
        conv_filter.w = weights
        image = jn.array([[[[2., 1., 3., 4.], [5., 6., 7., 8.],
                            [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        # NCHW: Batch, Channels/Colors, Height, Width
        features = conv_filter(image)
        expected_features = jn.array([[[[24., 43., 50., 21.],
                                        [42., 79., 87., 36.],
                                        [68., 118., 128., 52.],
                                        [20., 31., 34., 11.]]]])
        self.assertEqual(features.shape, (1, 1, 4, 4))
        self.assertTrue(jn.array_equal(features, expected_features))

    def test_on_conv2d_dilation_padding_valid(self):
        """
        Pass an input through two-by-two convolution filter
        with one channel and test the shape and contents of the output.
        """

        # Channels/Colors, #filters, filter_size (square)
        conv_filter = objax.nn.Conv2D(1, 1, 2, dilations=2, padding=objax.ConvPadding.VALID)
        weights = objax.TrainVar(jn.array([[[[1., 2.], [3., 4.]]]]).transpose((2, 3, 1, 0)))
        conv_filter.w = weights
        image = jn.array([[[[2., 1., 3., 4.], [5., 6., 7., 8.],
                            [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        # NCHW: Batch, Channels/Colors, Height, Width
        features = conv_filter(image)
        expected_features = jn.array([[[[79., 87.],
                                        [118., 128.]]]])
        self.assertEqual(features.shape, (1, 1, 2, 2))
        self.assertTrue(jn.array_equal(features, expected_features))

    def test_on_conv2d_group(self):
        """
        Pass an input through two two-by-two convolution filters
        with two channels and test the shape and contents of the output.
        """

        # Channels/Colors, #filters, filter_size (square)
        conv_filter = objax.nn.Conv2D(2, 2, 2, groups=2, padding=objax.ConvPadding.VALID)
        weights = objax.TrainVar(jn.array([[[[1., 2.], [3., 4.]]], [[[1., 2.], [3., 4.]]]]).transpose((2, 3, 1, 0)))
        conv_filter.w = weights
        image = jn.array([[[[2., 1., 3., 4.], [5., 6., 7., 8.],
                            [9., 10., 11., 12.], [13., 14., 15., 16.]],
                           [[1., 2., 3., 4.], [5., 6., 7., 8.],
                            [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        # NCHW: Batch, Channels/Colors, Height, Width
        features = conv_filter(image)
        expected_features = jn.array([[[[43., 53., 64.],
                                        [84., 94., 104.],
                                        [124., 134., 144.]],
                                       [[44., 54., 64.],
                                        [84., 94., 104.],
                                        [124., 134., 144.]]]])
        self.assertEqual(features.shape, (1, 2, 3, 3))
        self.assertTrue(jn.array_equal(features, expected_features))


if __name__ == '__main__':
    unittest.main()
