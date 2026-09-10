import unittest
from shelf_audit.track_video import track_frames,topological_order


def detection(x):
    return {'bbox_xyxy':[x,0,x+30,40], 'detector_score':.9,
            'recognition':{'status':'candidate_match','predicted_class':'same_sku','matches':[]}}


class TrackingTests(unittest.TestCase):
    def test_same_sku_packages_stay_separate(self):
        frames=[dict(frame_index=i,time_seconds=i*.05,detections=[detection(i),detection(100+i)]) for i in range(8)]
        tracks,_=track_frames(frames)
        self.assertEqual(len(tracks),2)
        self.assertTrue(all(t['detection_count']==8 for t in tracks))

    def test_short_gap_retains_track(self):
        frames=[dict(frame_index=0,time_seconds=0,detections=[detection(0)]),dict(frame_index=1,time_seconds=.2,detections=[detection(1)])]
        self.assertEqual(len(track_frames(frames)[0]),1)

    def test_long_gap_does_not_assume_same_instance(self):
        frames=[dict(frame_index=0,time_seconds=0,detections=[detection(0)]),dict(frame_index=1,time_seconds=1,detections=[detection(0)])]
        self.assertEqual(len(track_frames(frames)[0]),2)

    def test_cycle_is_flagged(self):
        self.assertEqual(topological_order([1,2],[(1,2),(2,1)]),(None,True))

    def test_missing_relation_is_ambiguous(self):
        self.assertTrue(topological_order([1,2],[])[1])

    def test_order_comes_from_edges_not_ids(self):
        self.assertEqual(topological_order([1,2,3],[(3,1),(1,2)]),([3,1,2],False))
