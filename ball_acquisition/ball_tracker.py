"""Single-ball nearest-motion association; observed and predicted rows stay distinct."""
import math

class BallTracker:
    def __init__(self,max_jump_px=180,max_gap_frames=5):
        self.max_jump_px=max_jump_px; self.max_gap=max_gap_frames; self.last=None; self.velocity=(0.,0.); self.track_id=1
    def update(self,frame_no,timestamp_s,detections):
        # detections: iterable of (xyxy, confidence)
        options=[]
        for box,conf in detections:
            x=(box[0]+box[2])/2; y=(box[1]+box[3])/2
            dist=0 if self.last is None else math.hypot(x-self.last[1],y-self.last[2])
            if self.last is None or dist<=self.max_jump_px*max(1,frame_no-self.last[0]): options.append((conf,dist,x,y))
        if options:
            conf,_,x,y=max(options,key=lambda a:a[0]-.001*a[1])
            if self.last and 0<frame_no-self.last[0]<=self.max_gap:
                dt=timestamp_s-self.last[3]
                if dt>0: self.velocity=((x-self.last[1])/dt,(y-self.last[2])/dt)
            self.last=(frame_no,x,y,timestamp_s)
            return {"frame":frame_no,"timestamp_s":round(timestamp_s,3),"track_id":self.track_id,"x_px":round(x,2),"y_px":round(y,2),"confidence":round(float(conf),4),"observed":True}
        if self.last and 0<frame_no-self.last[0]<=self.max_gap:
            dt=timestamp_s-self.last[3]
            return {"frame":frame_no,"timestamp_s":round(timestamp_s,3),"track_id":self.track_id,"x_px":round(self.last[1]+self.velocity[0]*dt,2),"y_px":round(self.last[2]+self.velocity[1]*dt,2),"confidence":"","observed":False}
        return None
