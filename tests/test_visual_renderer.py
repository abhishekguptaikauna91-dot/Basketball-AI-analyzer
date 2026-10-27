import unittest
import numpy as np
from drawers.video_drawer import VideoRenderer

class RendererTests(unittest.TestCase):
    def test_event_notification_does_not_retrigger_for_repeated_event(self):
        renderer=VideoRenderer(fps=30)
        event={"frame":20,"timestamp_s":1.0,"status":"confirmed_candidate"}
        for frame_no in range(20,24):
            frame=np.zeros((480,854,3),dtype=np.uint8)
            renderer.render(frame,confirmed_event=event,timestamp_s=frame_no/30,frame_no=frame_no)
        self.assertEqual(renderer.event_serial,1)

    def test_renderer_keeps_frame_dimensions(self):
        frame=np.zeros((720,1280,3),dtype=np.uint8)
        result=VideoRenderer().render(frame,timestamp_s=12.5,frame_no=625,court_calibrated=False)
        self.assertEqual(result.shape,(720,1280,3))

    def test_ball_trail_uses_observations_and_skips_predictions(self):
        renderer=VideoRenderer(fps=30)
        frame=np.zeros((240,320,3),dtype=np.uint8)
        renderer.render(frame.copy(),ball_track={'frame':0,'x_px':10,'y_px':20,'observed':True},frame_no=0)
        renderer.render(frame.copy(),ball_track={'frame':1,'x_px':15,'y_px':20,'observed':False},frame_no=1)
        renderer.render(frame.copy(),ball_track={'frame':2,'x_px':30,'y_px':25,'observed':True},frame_no=2)
        self.assertEqual(list(renderer.trail),[(10,20),(30,25)])

if __name__=='__main__':unittest.main()
