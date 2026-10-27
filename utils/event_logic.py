"""Temporal deduplication for ball-in-basket detections."""
class BasketEventDetector:
    def __init__(self, threshold=.45, confirm_frames=2, gap_tolerance_s=.5, cooldown_s=2.0):
        self.threshold=threshold; self.confirm_frames=confirm_frames
        self.gap_tolerance_s=gap_tolerance_s; self.cooldown_s=cooldown_s
        self.active=[]; self.last_seen=None; self.last_event=-1e9

    def update(self,timestamp_s,detections,frame_no=None):
        scores=[float(score) for _,score in detections if float(score)>=self.threshold]
        finalized=None
        if self.active and self.last_seen is not None and timestamp_s-self.last_seen>self.gap_tolerance_s:
            finalized=self._finalize(timestamp_s)
        if scores:
            self.active.append((timestamp_s,max(scores),frame_no)); self.last_seen=timestamp_s
        return finalized

    def _finalize(self,now):
        if not self.active: return None
        window=self.active; self.active=[]
        # Any model-positive run is a candidate; mark one-frame cases uncertain.
        if now-self.last_event<self.cooldown_s: return None
        self.last_event=window[0][0]
        return {"timestamp_s":round(window[0][0],3),"frame":window[0][2],"confidence":round(max(s for _,s,_ in window),4),
                "status":"confirmed_candidate" if len(window)>=self.confirm_frames else "uncertain_candidate",
                "shooter_track_id":""}

    def flush(self): return self._finalize(self.last_seen or 0)
