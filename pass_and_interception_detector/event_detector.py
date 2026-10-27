"""Conservative possession-change events from sustained ball/player proximity."""
import math

class InteractionDetector:
    def __init__(self,proximity_px=75,confirm_frames=3):
        self.proximity=proximity_px; self.confirm=confirm_frames; self.near_counts={}; self.possessor=None; self.events=[]
    def update(self,frame_no,timestamp_s,ball,players):
        if not ball or not ball.get("observed"): return None
        bx,by=ball["x_px"],ball["y_px"]; nearest=None; best=self.proximity
        for p in players:
            d=math.hypot(bx-p["center_x_px"],by-p["foot_y_px"])
            if d<best: nearest=p; best=d
        current=nearest["track_id"] if nearest else None
        if current is None: self.near_counts.clear(); return None
        self.near_counts[current]=self.near_counts.get(current,0)+1
        if self.near_counts[current]<self.confirm or current==self.possessor: return None
        previous=self.possessor; self.possessor=current
        event={"frame":frame_no,"timestamp_s":round(timestamp_s,3),"event":"possible_possession_change" if previous is not None else "possible_possession","from_track_id":previous or "","to_track_id":current,"confidence":"low","status":"uncertain","basis":"sustained nearest-player proximity; not proof of pass or interception"}
        self.events.append(event); self.near_counts={current:self.confirm}; return event
