import cv2
from drawers.overlay_utils import text

def draw_baskets(frame,baskets,cfg):
    color=tuple(cfg.get("basket_bgr",[40,170,255])); h,w=frame.shape[:2]
    for box,score in baskets:
        if score<float(cfg.get("basket_min_confidence",.4)): continue
        x1,y1,x2,y2=map(int,box); lx=max(7,min(17,(x2-x1)//3)); ly=max(7,min(17,(y2-y1)//3))
        for x,y,sx,sy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            cv2.line(frame,(x,y),(x+sx*lx,y),color,2,cv2.LINE_AA);cv2.line(frame,(x,y),(x,y+sy*ly),color,2,cv2.LINE_AA)
        if cfg.get("show_confidence",True): text(frame,f"NET {score:.2f}",(max(0,x1),max(16,y1-5)),.4,color,1)

def draw_ball(frame,detections,track,trail,cfg):
    color=tuple(cfg.get("ball_bgr",[40,235,245])); n=len(trail)
    for i,(x,y) in enumerate(trail):
        fade=(i+1)/max(1,n); r=max(2,round(2+3*fade));
        cv2.circle(frame,(x,y),r,color,-1,cv2.LINE_AA)
    if track and track.get("observed",True):
        score=track.get("confidence","");cx=int(track["x_px"]);cy=int(track["y_px"]);r=12
        cv2.circle(frame,(cx,cy),r,color,2,cv2.LINE_AA);cv2.circle(frame,(cx,cy),3,(255,255,255),-1,cv2.LINE_AA)
        if cfg.get("show_confidence",True) and score!="":
            cv2.putText(frame,f"BALL {float(score):.2f}",(cx+12,max(16,cy-7)),cv2.FONT_HERSHEY_DUPLEX,.42,color,1,cv2.LINE_AA)
    if track and not track.get("observed",True):
        x,y=int(track["x_px"]),int(track["y_px"]);cv2.circle(frame,(x,y),12,(180,190,200),1,cv2.LINE_AA);text(frame,"PREDICTED",(x+12,y),.35,(190,200,210),1)
