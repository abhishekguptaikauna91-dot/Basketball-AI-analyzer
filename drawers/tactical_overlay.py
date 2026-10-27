import cv2
from drawers.overlay_utils import panel,text
from tactical_view_converter.tactical_view import render_court

def draw_tactical_status(frame,calibrated,cfg,players=None,ball=None):
    if not cfg.get("show_court_status",True):return
    if calibrated and cfg.get("mini_court_enabled",True) and players is not None:
        h,w=frame.shape[:2]; setting=cfg.get("mini_court_position",[0,0]); x,y=map(int,setting)
        mw,mh=map(int,cfg.get("mini_court_size",[230,128]))
        if x==0:x=w-mw-20
        if y==0:y=20
        panel(frame,(x-5,y-5,x+mw+5,y+mh+5),alpha=.82,radius=9)
        court=render_court(players,ball,width=mw,height=mh)
        frame[y:y+mh,x:x+mw]=court
        return
    h,w=frame.shape[:2]; x,y=map(int,cfg.get("court_status_position",[20,0])); y=y or h-43
    panel(frame,(x,y,x+232,y+28),alpha=.72,radius=7)
    label="COURT MAP  /  UNCALIBRATED" if not calibrated else "COURT MAP  /  POSITIONS UNAVAILABLE"
    text(frame,label,(x+9,y+19),.36,(180,195,207),1)
