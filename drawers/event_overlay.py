import math
from drawers.overlay_utils import panel,text

def draw_event(frame,event,age,cfg,secondary=False):
    h,w=frame.shape[:2];duration=float(cfg.get("event_duration_seconds",2.2));fade=min(1.,max(0.,(duration-age)/.4),min(1.,age/.15+0.2))
    if secondary:
        title="POSSIBLE POSSESSION CHANGE";sub="LOW CONFIDENCE  ·  PROXIMITY SIGNAL"
        x,y=20,h-102
    else:
        title="BUCKET!" if event.get("status")=="confirmed_candidate" else "BASKET CANDIDATE"
        team=event.get("scoring_team") or "TEAM UNASSIGNED"
        sub=f"{event.get('timestamp_s',0):05.2f}s  ·  {team}  ·  PLAYER UNASSIGNED"
        x,y=max(20,w-340),max(100,int(h*.17))
    pulse=1+.025*math.sin(age*8); width=300 if not secondary else 350; height=72 if not secondary else 58
    panel(frame,(x,y,x+int(width*pulse),y+height),(18,30,45),alpha=.78*fade,radius=12,border=(50,180,210))
    text(frame,title,(x+16,y+28),.68 if not secondary else .40,(95,230,242),2 if not secondary else 1)
    text(frame,sub,(x+16,y+52),.37,(225,232,238),1)
