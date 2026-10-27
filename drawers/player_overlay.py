import cv2
from drawers.overlay_utils import panel,text

def _brackets(frame,box,color,length,thick):
    x1,y1,x2,y2=map(int,box); l=min(length,max(8,(x2-x1)//3,(y2-y1)//5));
    for x,y,sx,sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
        cv2.line(frame,(x,y),(x+sx*l,y),color,thick,cv2.LINE_AA);cv2.line(frame,(x,y),(x,y+sy*l),color,thick,cv2.LINE_AA)

def draw_players(frame,players,assignments,cfg):
    h,w=frame.shape[:2]; colors=cfg.get("team_colors",{}); placed=[]
    boxes=[(int(p["x1"]),int(p["y1"]),int(p["x2"]),int(p["y2"])) for p in players]
    for index,p in enumerate(players):
        tid=p.get("track_id",""); team=assignments.get(tid,{}) if isinstance(assignments,dict) else {}
        team_name=team.get("team") if team.get("confirmed") else None
        color=tuple(colors.get(team_name,[75,220,150])) if team_name else tuple(cfg.get("unknown_player_bgr",[75,220,150]))
        box=[p["x1"],p["y1"],p["x2"],p["y2"]];_brackets(frame,box,color,int(cfg.get("bracket_length",13)),int(cfg.get("player_line_thickness",2)))
        identity=(f"TRACK ID {tid}" if tid!="" else "PLAYER") if cfg.get("show_ids",True) else "PLAYER"
        label=(team_name+"  ·  " if team_name else "")+identity
        pad=int(cfg.get("label_padding",8)); tw,_=cv2.getTextSize(label,cv2.FONT_HERSHEY_DUPLEX,float(cfg.get("label_scale",.43)),1)[0]; bw=min(w,tw+2*pad); bh=int(cfg.get("label_height",27))
        bx1,by1,bx2,by2=map(int,box); base_x=max(4,min(w-bw-4,bx1)); above=by1-bh-5; below=by2+5
        candidates=[(base_x,above),(base_x,below),(bx1-bw-6,by1),(bx2+6,by1),
                    (bx2-bw,above),(bx2-bw,below)]
        for row in range(2,6): candidates.extend([(base_x,by1-row*(bh+4)),(base_x,by2+(row-1)*(bh+4))])
        other_boxes=[b for j,b in enumerate(boxes) if j!=index]
        def overlaps(a,b): return a[0]<b[2]+4 and a[2]+4>b[0] and a[1]<b[3]+4 and a[3]+4>b[1]
        safe_bottom=int(h*float(cfg.get("safe_bottom_ratio",.14)))
        def valid(cx,cy):
            rect=(cx,cy,cx+bw,cy+bh)
            return cx>=4 and cy>=4 and rect[2]<w-4 and rect[3]<h-safe_bottom and not any(overlaps(rect,r) for r in placed) and not any(overlaps(rect,r) for r in other_boxes)
        pos=next(((cx,cy) for cx,cy in candidates if valid(cx,cy)),None)
        if pos is None: pos=(base_x,max(4,min(h-bh-safe_bottom,above)))
        x,y=pos; placed.append((x,y,x+bw,y+bh)); panel(frame,(x,y,x+bw,y+bh),color=color,alpha=float(cfg.get("panel_alpha",.84)),radius=7,border=color)
        text(frame,label,(x+pad,y+int(bh*.68)),float(cfg.get("label_scale",.43)),tuple(cfg.get("text_bgr",[245,248,252])),1)
