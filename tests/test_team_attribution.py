import unittest
from shot_detector.team_attributor import ScoringTeamAttributor

class TeamAttributionTests(unittest.TestCase):
    def test_assigns_team_only_after_repeated_clear_ball_contact(self):
        event={'frame':6,'timestamp_s':.6}
        balls=[{'frame':f,'timestamp_s':f/10,'x_px':110,'y_px':110,'observed':True} for f in (3,4,5)]
        players=[[{'frame':f,'track_id':7,'center_x_px':100,'center_y_px':100,'foot_y_px':130}] for f in (3,4,5)]
        teams={7:{'team':'Team A','confidence':.85,'confirmed':True}}
        result=ScoringTeamAttributor().attribute(event,balls,players,teams)
        self.assertEqual(result['scoring_team'],'Team A')
        self.assertEqual(result['associated_track_id'],7)

    def test_leaves_team_unassigned_without_enough_ball_observations(self):
        event={'frame':6,'timestamp_s':.6}
        result=ScoringTeamAttributor().attribute(event,[],[],{})
        self.assertEqual(result['scoring_team'],'')
        self.assertEqual(result['associated_track_id'],'')

if __name__=='__main__':unittest.main()
