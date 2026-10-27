"""Original broadcast-style renderer. Consumes pipeline results; performs no inference."""
from collections import defaultdict, deque
from pathlib import Path
import cv2
import yaml

from drawers.player_overlay import draw_players
from drawers.ball_overlay import draw_ball, draw_baskets
from drawers.scoreboard import draw_scoreboard
from drawers.event_overlay import draw_event
from drawers.tactical_overlay import draw_tactical_status

class VideoRenderer:
    def __init__(self, config_path=None, fps=30):
        config_path=Path(config_path) if config_path else Path(__file__).resolve().parents[1]/"configs/visual_config.yaml"
        self.cfg=yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        trail_len=int(self.cfg.get("ball_trail_length",18))
        if self.cfg.get("lightweight_mode",False): trail_len=min(7,trail_len)
        self.trail=deque(maxlen=trail_len)
        self.player_trails=defaultdict(lambda:deque(maxlen=int(self.cfg.get("player_trail_length",14))))
        self.player_last_seen={}
        self.fps=fps; self.event_start=None; self.event=None; self.event_serial=0; self._seen_events=set()

    def render(self, frame, *, players=(), ball_detections=(), ball_track=None, baskets=(),
               confirmed_event=None, team_assignments=None, timestamp_s=0., frame_no=0,
               court_calibrated=False, court_positions=None, ball_court_position=None,
               player_stats=None, team_makes=None, unassigned_makes=0, interaction_event=None):
        cfg=self.cfg; assignments=team_assignments or {}
        if cfg.get("show_player_trails",True) and not cfg.get("lightweight_mode",False):
            for player in players:
                tid=player.get("track_id")
                if tid in (None,""): continue
                point=(int(player["center_x_px"]),int(player["foot_y_px"]))
                self.player_trails[tid].append(point); self.player_last_seen[tid]=frame_no
            for tid in list(self.player_trails):
                if frame_no-self.player_last_seen.get(tid,frame_no)>max(1,int(self.fps*1.5)):
                    del self.player_trails[tid]; self.player_last_seen.pop(tid,None)
            for tid,points in self.player_trails.items():
                state=assignments.get(tid,{})
                team=state.get("team") if state.get("confirmed") else None
                color=tuple(cfg.get("team_colors",{}).get(team,cfg.get("trail_bgr",[120,170,175])))
                points=list(points)
                for i,(a,b) in enumerate(zip(points,points[1:])):
                    alpha=(i+1)/max(1,len(points)-1)
                    tint=tuple(int(45+channel*.38*alpha) for channel in color)
                    cv2.line(frame,a,b,tint,1,cv2.LINE_AA)
        if ball_track and ball_track.get("observed"):
            self.trail.append((int(ball_track["x_px"]),int(ball_track["y_px"])))
        if ball_track is None: self.trail.clear()
        if cfg.get("show_baskets",True): draw_baskets(frame,baskets,cfg)
        if cfg.get("show_players",True): draw_players(frame,players,assignments,cfg)
        if cfg.get("show_ball",True): draw_ball(frame,ball_detections,ball_track,list(self.trail),cfg)
        if confirmed_event is not None:
            key=(confirmed_event.get("frame"),confirmed_event.get("timestamp_s"),confirmed_event.get("status"))
            if key not in self._seen_events:
                self._seen_events.add(key)
                self.event=confirmed_event; self.event_start=timestamp_s; self.event_serial+=1
        draw_scoreboard(frame,timestamp_s,team_makes,unassigned_makes,cfg)
        if self.event is not None and self.event_start is not None:
            age=timestamp_s-self.event_start
            if age<=float(cfg.get("event_duration_seconds",2.2)) and cfg.get("show_events",True):
                draw_event(frame,self.event,age,cfg)
            else:
                self.event=None; self.event_start=None
        if interaction_event and cfg.get("show_events",True):
            draw_event(frame,interaction_event,0,cfg,secondary=True)
        if cfg.get("show_court_status",True): draw_tactical_status(frame,court_calibrated,cfg,players=court_positions,ball=ball_court_position)
        return frame
