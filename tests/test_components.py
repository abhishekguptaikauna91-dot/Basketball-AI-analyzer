import unittest
import numpy as np
from team_assigner.team_assigner import TeamAssigner
from ball_acquisition.ball_tracker import BallTracker
from speed_and_distance_calculator.movement_calculator import calculate_tracks

class ComponentTests(unittest.TestCase):
    def test_uniform_clustering_confirms_two_visual_groups(self):
        a=np.zeros((100,80,3),dtype=np.uint8); a[:]=(180,20,20)
        b=np.zeros((100,80,3),dtype=np.uint8); b[:]=(20,20,180)
        assigner=TeamAssigner(min_samples=3)
        for _ in range(4):
            assigner.observe(1,a,[0,0,80,100]); assigner.observe(2,b,[0,0,80,100])
        result=assigner.finalize()
        self.assertEqual(set(v['team'] for v in result.values()),{'Team A','Team B'})
        self.assertTrue(all(v['confirmed'] for v in result.values()))

    def test_ball_missing_prediction_is_marked_estimated(self):
        tracker=BallTracker(max_gap_frames=2)
        first=tracker.update(0,0.0,[([0,0,10,10],.9)])
        second=tracker.update(1,.1,[([10,0,20,10],.9)])
        missing=tracker.update(2,.2,[])
        self.assertTrue(first['observed']); self.assertTrue(second['observed'])
        self.assertFalse(missing['observed'])
        self.assertIsNone(tracker.update(5,.5,[]))

    def test_movement_uses_world_units_and_time(self):
        rows=[{'track_id':2,'timestamp_s':0.,'x_m':0.,'y_m':0.}, {'track_id':2,'timestamp_s':2.,'x_m':6.,'y_m':8.}]
        result=calculate_tracks(rows)[0]
        self.assertEqual(result['distance_m'],10.)
        self.assertEqual(result['average_speed_m_s'],5.)

if __name__=='__main__': unittest.main()
