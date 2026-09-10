import unittest
import torch
from shelf_audit.detector import match_boxes


class DetectorTests(unittest.TestCase):
    def test_duplicate_is_false_positive(self):
        box = torch.tensor([[0., 0., 10., 10.]])
        self.assertEqual(match_boxes(box.repeat(2,1), box), (1,1,0))

    def test_empty_predictions_are_misses(self):
        self.assertEqual(match_boxes(torch.empty((0,4)), torch.tensor([[0.,0.,10.,10.]])), (0,0,1))

    def test_wrong_location_is_not_a_match(self):
        self.assertEqual(match_boxes(torch.tensor([[20.,20.,30.,30.]]), torch.tensor([[0.,0.,10.,10.]])), (0,1,1))

    def test_empty_truth_is_false_positive(self):
        self.assertEqual(match_boxes(torch.tensor([[0.,0.,10.,10.]]), torch.empty((0,4))), (0,1,0))


if __name__ == '__main__':
    unittest.main()
