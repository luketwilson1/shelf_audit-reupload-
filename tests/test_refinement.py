import unittest
from shelf_audit.refine_predictions import clean_frame
from shelf_audit.track_video import track_frames


def d(box,accepted=True,sku='a'):
    return dict(bbox_xyxy=box,recognition=dict(status='candidate_match' if accepted else 'review_needed',predicted_class=sku if accepted else None))


class RefinementTests(unittest.TestCase):
    def test_fragment_retained_in_audit(self):
        result=clean_frame(dict(detections=[d([0,0,100,100]),d([5,5,20,90],False)]))
        self.assertEqual(len(result['detections']),1)
        self.assertEqual(len(result['excluded_detections']),1)

    def test_two_accepted_same_sku_stay(self):
        self.assertEqual(len(clean_frame(dict(detections=[d([0,0,100,100]),d([5,5,20,90])]))['detections']),2)

    def test_merged_box_needs_two_explanations(self):
        ds=[d([0,0,100,100],False),d([0,0,49,100]),d([51,0,100,100],sku='b')]
        self.assertEqual(len(clean_frame(dict(detections=ds))['detections']),2)

    def test_isolated_unknown_remains(self):
        self.assertEqual(len(clean_frame(dict(detections=[d([0,0,100,100],False)]))['detections']),1)

    def test_reconnection_requires_matching_identity_and_space(self):
        frames=[dict(frame_index=i,time_seconds=i*.05,detections=[d([0,0,100,100])]) for i in range(4)]
        frames.append(dict(frame_index=4,time_seconds=.6,detections=[d([2,0,102,100])]))
        self.assertEqual(len(track_frames(frames,reconnect=True)[0]),1)
        frames[-1]['detections']=[d([2,0,102,100],sku='b')]
        self.assertEqual(len(track_frames(frames,reconnect=True)[0]),2)
