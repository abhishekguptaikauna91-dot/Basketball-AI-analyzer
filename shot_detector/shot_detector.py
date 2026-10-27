"""Basket event wrapper; attempts and shooter attribution remain separate evidence."""
from utils.event_logic import BasketEventDetector

class ShotDetector(BasketEventDetector):
    def update_frame(self, frame_no, timestamp_s, detections, ball_observed=False):
        event=self.update(timestamp_s,detections,frame_no)
        if event:
            event["evidence"]="ball_in_basket detection" + ("; ball observed nearby in frame" if ball_observed else "")
        return event
