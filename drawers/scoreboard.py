from drawers.overlay_utils import panel,text

def draw_scoreboard(frame,timestamp,team_makes=None,unassigned_makes=0,cfg=None):
    cfg=cfg or {}; team_makes=team_makes or {}
    if not cfg.get("show_scoreboard",True): return
    h,w=frame.shape[:2]; x,y=map(int,cfg.get("scoreboard_position",[20,20])); scale=min(1.25,max(.72,w/1600)); width=int(570*scale); height=int(68*scale)
    panel(frame,(x,y,x+width,y+height),alpha=float(cfg.get("panel_alpha",.84)),radius=int(12*scale))
    text(frame,"COURTSIDE  /  AI ANALYSIS",(x+16*scale,y+24*scale),.48*scale,(95,221,239),1)
    text(frame,f"VIDEO  {int(timestamp//60):02d}:{timestamp%60:04.1f}",(x+16*scale,y+50*scale),.46*scale)
    names=cfg.get("team_names",{})
    name_a=names.get("Team A","TEAM A"); name_b=names.get("Team B","TEAM B")
    count_a=int(team_makes.get("Team A",0)); count_b=int(team_makes.get("Team B",0))
    text(frame,f"{name_a}  MADE {count_a}  PTS —",(x+239*scale,y+29*scale),.46*scale,tuple(cfg.get("team_colors",{}).get("Team A",[90,190,255])),1)
    text(frame,f"{name_b}  MADE {count_b}  PTS —",(x+239*scale,y+52*scale),.46*scale,tuple(cfg.get("team_colors",{}).get("Team B",[255,175,85])),1)
    text(frame,f"UNASSIGNED MAKES {int(unassigned_makes)}",(x+16*scale,y+50*scale),.35*scale,tuple(cfg.get("accent_bgr",[35,205,225])),1)
