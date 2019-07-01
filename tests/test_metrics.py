import unittest
import torchrain as tr
import numpy as np


class TestMetrics(unittest.TestCase):
    max_batch = 20
    max_samples = 50
    min_samples = 10
    min_batch = 1
    n_class = 23

    def test_accuracy_single_class(self):
        batch_size = np.round(TestMetrics.max_batch * np.random.rand(1)).astype('int')[0] + TestMetrics.min_batch
        sample_size = np.round(TestMetrics.max_samples * np.random.rand(1)).astype('int')[0] + TestMetrics.min_samples
        one_vector = np.ones([batch_size, sample_size])
        zero_vector = np.zeros([batch_size, sample_size])

        self.assertTrue(tr.metrics.accuracy(zero_vector, zero_vector) == 1)
        self.assertTrue(tr.metrics.accuracy(one_vector, zero_vector) == 0)
        self.assertTrue(tr.metrics.accuracy(zero_vector, one_vector) == 0)
        self.assertTrue(tr.metrics.accuracy(one_vector, one_vector) == 1)

    def test_accuracy_multi_class_class(self):
        batch_size = np.round(TestMetrics.max_batch * np.random.rand(1)).astype('int')[0] + TestMetrics.min_batch
        sample_size = np.round(TestMetrics.max_samples * np.random.rand(1)).astype('int')[0] + TestMetrics.min_samples
        one_vector = np.zeros([batch_size, sample_size, TestMetrics.n_class])
        zero_vector = np.zeros([batch_size, sample_size, TestMetrics.n_class])
        for i in range(batch_size):
            for j in range(sample_size):
                k = np.round(TestMetrics.n_class * np.random.randint(1)).astype('int')
                one_vector[i, j, k] = 1
                zero_vector[i, j, (k + 1) % TestMetrics.n_class] = 1
        self.assertTrue(tr.metrics.accuracy(one_vector, one_vector) == 1)
        self.assertTrue(tr.metrics.accuracy(zero_vector, zero_vector) == 1)
        self.assertTrue(tr.metrics.accuracy(one_vector, zero_vector) == 0)
        self.assertTrue(tr.metrics.accuracy(zero_vector, one_vector) == 0)

    def test_classification_exception(self):

        one_vector = np.zeros([TestMetrics.n_class])
        with self.assertRaises(Exception) as context:
            tr.metrics.accuracy(one_vector, one_vector)
        self.assertTrue('Input arrays must have 2 or 3 dimension' == str(context.exception))

    def test_mse(self):
        x = np.random.randn(10, 10)
        self.assertEqual(tr.metrics.mse(x, x), 0)

    def test_nmse(self):
        x = np.random.randn(10, 10)
        self.assertEqual(tr.metrics.nmse(x, x), 0)
