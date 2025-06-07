import unittest
import numpy as np
from src.eval.metrics.loss import avg_loss, max_loss


class TestLoss(unittest.TestCase):
    def setUp(self):
        # Create a simple adjacency matrix for testing
        self.adj_matrix = np.array([
            [0, 1, 2, 3],
            [1, 0, 4, 5],
            [2, 4, 0, 6],
            [3, 5, 6, 0]
        ])

    def test_avg_loss(self):
        # Test average loss for agent 0 in cluster [1, 2]
        self.assertEqual(avg_loss(0, [1, 2], self.adj_matrix), 1.5)
        # Test average loss for agent 1 in cluster [0, 2, 3]
        self.assertEqual(avg_loss(1, [0, 2, 3], self.adj_matrix), np.float64(10/3))
        # Test average loss for single node cluster
        self.assertEqual(avg_loss(0, [1], self.adj_matrix), 1.0)

    def test_max_loss(self):
        # Test max loss for agent 0 in cluster [1, 2]
        self.assertEqual(max_loss(0, [1, 2], self.adj_matrix), 2.0)
        # Test max loss for agent 1 in cluster [0, 2, 3]
        self.assertEqual(max_loss(1, [0, 2, 3], self.adj_matrix), 5.0)
        # Test max loss for single node cluster
        self.assertEqual(max_loss(0, [1], self.adj_matrix), 1.0)


if __name__ == '__main__':
    unittest.main()
