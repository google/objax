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


class TestConvTranspose(unittest.TestCase):
    def test_on_conv_transpose_2d_three_by_three(self):
        """
        Pass an input through a three-by-three convolution filter and
        test the shape and contents of the output.
        """
        w_init = lambda s: jn.array([[[[1., 2., 1.], [1., 2., 1.], [1., 1., 1.]]]]).transpose((2, 3, 0, 1))
        conv = objax.nn.ConvTranspose2D(1, 1, 3, padding=objax.ConvPadding.VALID, w_init=w_init)
        x = jn.array([[[[2., 1., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        y = jn.array([[[[2., 5., 7., 11., 11., 4.],
                        [7., 21., 31., 39., 34., 12.],
                        [16., 47., 70., 80., 65., 24.],
                        [27., 79., 114., 125., 97., 36.],
                        [22., 59., 86., 93., 70., 28.],
                        [13., 27., 42., 45., 31., 16.]]]])
        self.assertEqual(conv(x).tolist(), y.tolist())

    def test_on_conv_transpose_2d_two_by_two(self):
        """
        Pass an input through a two-by-two convolution filter and
        test the shape and contents of the output.
        """
        w_init = lambda s: jn.array([[[[1., 2.], [3., 4.]]]]).transpose((2, 3, 0, 1))
        conv = objax.nn.ConvTranspose2D(1, 1, 2, padding=objax.ConvPadding.VALID, w_init=w_init)
        x = jn.array([[[[2., 1., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        y = jn.array([[[[2., 5., 5., 10., 8.],
                        [11., 27., 32., 46., 32.],
                        [24., 66., 76., 86., 56.],
                        [40., 106., 116., 126., 80.],
                        [39., 94., 101., 108., 64.]]]])
        self.assertEqual(conv(x).tolist(), y.tolist())

    def test_on_conv_transpose_2d_padding(self):
        """
        Pass an input through a two-by-two convolution filter with padding and
        test the shape and contents of the output.
        """
        x = jn.array([[[[2., 1., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        y = jn.array([[[[2., 5., 5., 10.], [11., 27., 32., 46.], [24., 66., 76., 86.], [40., 106., 116., 126.]]]])
        w_init = lambda s: jn.array([[[[1., 2.], [3., 4.]]]]).transpose((2, 3, 0, 1))
        conv = objax.nn.ConvTranspose2D(1, 1, 2, padding=objax.ConvPadding.SAME, w_init=w_init)
        self.assertEqual(conv(x).tolist(), y.tolist())
        conv = objax.nn.ConvTranspose2D(1, 1, 2, padding='same', w_init=w_init)
        self.assertEqual(conv(x).tolist(), y.tolist())
        conv = objax.nn.ConvTranspose2D(1, 1, 2, padding='Same', w_init=w_init)
        self.assertEqual(conv(x).tolist(), y.tolist())
        conv = objax.nn.ConvTranspose2D(1, 1, 2, padding='SAME', w_init=w_init)
        self.assertEqual(conv(x).tolist(), y.tolist())
        conv = objax.nn.ConvTranspose2D(1, 1, 2, padding=(1, 0), w_init=w_init)
        self.assertEqual(conv(x).tolist(), y.tolist())
        conv = objax.nn.ConvTranspose2D(1, 1, 2, padding=[(1, 0), (1, 0)], w_init=w_init)
        self.assertEqual(conv(x).tolist(), y.tolist())
        y = [[[[2., 5., 5., 10., 8.], [11., 27., 32., 46., 32.], [24., 66., 76., 86., 56.],
               [40., 106., 116., 126., 80.], [39., 94., 101., 108., 64.]]]]
        conv = objax.nn.ConvTranspose2D(1, 1, 2, padding=1, w_init=w_init)
        self.assertEqual(conv(x).tolist(), y)
        conv = objax.nn.ConvTranspose2D(1, 1, 2, padding=(1, 1), w_init=w_init)
        self.assertEqual(conv(x).tolist(), y)
        conv = objax.nn.ConvTranspose2D(1, 1, 2, padding=[(1, 1), (1, 1)], w_init=w_init)
        self.assertEqual(conv(x).tolist(), y)

    def test_on_conv_transpose_2d_stride(self):
        """
        Pass an input through a two-by-two convolution filter with stride=2
        and test the shape and contents of the output.
        """

        # Channels/Colors, #filters, filter_size (square)
        conv_filter = objax.nn.ConvTranspose2D(1, 1, 2, strides=2, padding=objax.ConvPadding.VALID)
        weights = objax.TrainVar(jn.array([[[[1., 2.], [3., 4.]]]]).transpose((2, 3, 0, 1)))
        conv_filter.w = weights
        image = jn.array([[[[2., 1., 3., 4.],
                            [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        # NCHW: Batch, Channels/Colors, Height, Width
        features = conv_filter(image)
        expected_features = jn.array([[[[2., 4., 1., 2., 3., 6., 4., 8.],
                                        [6., 8., 3., 4., 9., 12., 12., 16.],
                                        [5., 10., 6., 12., 7., 14., 8., 16.],
                                        [15., 20., 18., 24., 21., 28., 24., 32.],
                                        [9., 18., 10., 20., 11., 22., 12., 24.],
                                        [27., 36., 30., 40., 33., 44., 36., 48.],
                                        [13., 26., 14., 28., 15., 30., 16., 32.],
                                        [39., 52., 42., 56., 45., 60., 48., 64.]]]])
        self.assertEqual(features.shape, (1, 1, 8, 8))
        self.assertTrue(jn.array_equal(features, expected_features))

    def test_on_conv_transpose_2d_two_filters(self):
        """
        Pass an input through two two-by-two convolution filter
        and test the shape and contents of the output.
        """

        # Channels/Colors, #filters, filter_size (square)
        conv_filter = objax.nn.ConvTranspose2D(1, 2, 2, padding=objax.ConvPadding.VALID)
        weights = objax.TrainVar(jn.array([[[[1., 2.], [3., 4.]]], [[[1., 2.], [3., 4.]]]]).transpose((2, 3, 0, 1)))
        conv_filter.w = weights
        image = jn.array([[[[2., 1., 3., 4.],
                            [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        # NCHW: Batch, Channels/Colors, Height, Width
        features = conv_filter(image)
        expected_features = jn.array([[[[2., 5., 5., 10., 8.],
                                        [11., 27., 32., 46., 32.],
                                        [24., 66., 76., 86., 56.],
                                        [40., 106., 116., 126., 80.],
                                        [39., 94., 101., 108., 64.]],
                                       [[2., 5., 5., 10., 8.],
                                        [11., 27., 32., 46., 32.],
                                        [24., 66., 76., 86., 56.],
                                        [40., 106., 116., 126., 80.],
                                        [39., 94., 101., 108., 64.]]]])
        self.assertEqual(features.shape, (1, 2, 5, 5))
        self.assertTrue(jn.array_equal(features, expected_features))

    def test_on_conv_transpose_2d_two_channels(self):
        """
        Pass an input through two-by-two convolution filter
        with two channels and test the shape and contents of the output.
        """

        # Channels/Colors, #filters, filter_size (square)
        conv_filter = objax.nn.ConvTranspose2D(2, 1, 2, padding=objax.ConvPadding.VALID)
        weights = objax.TrainVar(jn.array([[[[1., 2.], [3., 4.]], [[1., 2.], [3., 4.]]]]).transpose((2, 3, 0, 1)))
        conv_filter.w = weights
        image = jn.array([[[[2., 1., 3., 4.], [5., 6., 7., 8.],
                            [9., 10., 11., 12.], [13., 14., 15., 16.]],
                           [[2., 1., 3., 4.], [5., 6., 7., 8.],
                            [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        # NCHW: Batch, Channels/Colors, Height, Width
        features = conv_filter(image)
        expected_features = jn.array([[[[4., 10., 10., 20., 16.],
                                        [22., 54., 64., 92., 64.],
                                        [48., 132., 152., 172., 112.],
                                        [80., 212., 232., 252., 160.],
                                        [78., 188., 202., 216., 128.]]]])
        self.assertEqual(features.shape, (1, 1, 5, 5))
        self.assertTrue(jn.array_equal(features, expected_features))

    def test_on_conv_transpose_2d_dilation_padding_same(self):
        """
        Pass an input through two-by-two convolution filter
        with one channel and test the shape and contents of the output.
        """

        # Channels/Colors, #filters, filter_size (square)
        conv_filter = objax.nn.ConvTranspose2D(1, 1, 2, dilations=2, padding=objax.ConvPadding.SAME)
        weights = objax.TrainVar(jn.array([[[[1., 2.], [3., 4.]]]]).transpose((2, 3, 1, 0)))
        conv_filter.w = weights
        image = jn.array([[[[2., 1., 3., 4.], [5., 6., 7., 8.],
                            [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        # NCHW: Batch, Channels/Colors, Height, Width
        features = conv_filter(image)
        expected_features = jn.array([[[[6., 17., 20., 14.],
                                        [13., 46., 48., 34.],
                                        [32., 82., 92., 58.],
                                        [30., 69., 76., 44.]]]])
        self.assertEqual(features.shape, (1, 1, 4, 4))
        self.assertTrue(jn.array_equal(features, expected_features))

    def test_on_conv_transpose_2d_dilation_padding_valid(self):
        """
        Pass an input through two-by-two convolution filter
        with one channel and test the shape and contents of the output.
        """

        # Channels/Colors, #filters, filter_size (square)
        conv_filter = objax.nn.ConvTranspose2D(1, 1, 2, dilations=2, padding=objax.ConvPadding.VALID)
        weights = objax.TrainVar(jn.array([[[[1., 2.], [3., 4.]]]]).transpose((2, 3, 1, 0)))
        conv_filter.w = weights
        image = jn.array([[[[2., 1., 3., 4.], [5., 6., 7., 8.],
                            [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
        # NCHW: Batch, Channels/Colors, Height, Width
        features = conv_filter(image)
        expected_features = jn.array([[[[2., 1., 7., 6., 6., 8.],
                                        [5., 6., 17., 20., 14., 16.],
                                        [15., 13., 46., 48., 34., 40.],
                                        [28., 32., 82., 92., 58., 64.],
                                        [27., 30., 69., 76., 44., 48.],
                                        [39., 42., 97., 104., 60., 64.]]]])
        self.assertEqual(features.shape, (1, 1, 6, 6))
        self.assertTrue(jn.array_equal(features, expected_features))


if __name__ == '__main__':
    unittest.main()
