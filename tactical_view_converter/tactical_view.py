"""Top-down plotting for coordinates only after a valid court mapping exists."""
import cv2

def render_court(points,players=(),ball=None,width=600,height=320):
    """Points are normalized court coordinates in metres (94x50 court)."""
    image=255*__import__('numpy').ones((height,width,3),dtype='uint8'); sx=width/94; sy=height/50
    cv2.rectangle(image,(1,1),(width-2,height-2),(40,40,40),2); cv2.line(image,(width//2,0),(width//2,height),(40,40,40),2)
    cv2.circle(image,(width//2,height//2),int(6*sx),(40,40,40),2)
    for p in players:
        x=int(p["x_m"]*sx); y=int(p["y_m"]*sy); cv2.circle(image,(x,y),7,p.get("color",(0,120,255)),-1)
    if ball: cv2.circle(image,(int(ball["x_m"]*sx),int(ball["y_m"]*sy)),5,(0,0,220),-1)
    return image
