import unittest
import numpy as np
from shelf_audit.recognizer import rank_matches

class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.records=[dict(class_id=c,product=c,crop_file=str(i)) for i,c in enumerate(['a','a','b','c'])]
        self.v=np.array([[1,0],[.8,.6],[0,1],[-1,0]],dtype=np.float32)
    def test_distinct_product_ranking(self):
        r=rank_matches(np.array([1,0]),self.v,self.records)
        self.assertEqual([m['class_id'] for m in r['matches']],['a','b','c'])
        self.assertEqual(r['predicted_class'],'a')
    def test_exclusion_removes_self(self):
        r=rank_matches(np.array([1,0]),self.v,self.records,exclude=[0])
        self.assertEqual(r['matches'][0]['reference_index'],1)
        self.assertAlmostEqual(r['matches'][0]['similarity'],.8,places=5)
    def test_ambiguous_and_low_match_rejected(self):
        r=rank_matches(np.array([1,1]),self.v,self.records,exclude=[1],min_score=.9)
        self.assertIsNone(r['predicted_class'])
        self.assertIn('low_similarity',r['reasons'])
        self.assertIn('ambiguous_match',r['reasons'])
    def test_invalid_thresholds(self):
        with self.assertRaises(ValueError):rank_matches(np.array([1,0]),self.v,self.records,min_score=2)
    def test_no_candidates(self):
        with self.assertRaises(ValueError):rank_matches(np.array([1,0]),self.v,self.records,exclude=range(4))

if __name__=='__main__':unittest.main()
