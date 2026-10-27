import cv2

def panel(frame,rect,color=(15,24,37),alpha=.82,radius=12,border=(72,95,115)):
    x1,y1,x2,y2=map(int,rect); h,w=frame.shape[:2]
    x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
    if x2<=x1 or y2<=y1:return
    layer=frame.copy(); cv2.rectangle(layer,(x1+radius,y1),(x2-radius,y2),color,-1)
    cv2.rectangle(layer,(x1,y1+radius),(x2,y2-radius),color,-1)
    for pt in [(x1+radius,y1+radius),(x2-radius,y1+radius),(x1+radius,y2-radius),(x2-radius,y2-radius)]:cv2.circle(layer,pt,radius,color,-1)
    cv2.addWeighted(layer,alpha,frame,1-alpha,0,frame)
    cv2.rectangle(frame,(x1,y1),(x2,y2),border,1,cv2.LINE_AA)

def text(frame,value,xy,scale=.55,color=(240,245,250),thickness=1):
    cv2.putText(frame,str(value),tuple(map(int,xy)),cv2.FONT_HERSHEY_DUPLEX,scale,color,thickness,cv2.LINE_AA)
