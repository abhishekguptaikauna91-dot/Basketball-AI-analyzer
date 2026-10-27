"""Player detection with BoT-SORT when installed and a centroid fallback."""
from types import SimpleNamespace
import numpy as np
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path):
        self.model = YOLO(str(model_path)); self.names = self.model.names
        if not self.names or "player" not in [str(v).lower() for v in self.names.values()]:
            raise ValueError(f"Expected a player class, found {self.names}")
        try:
            import lap  # noqa: F401
            self._tracker_failed=False
        except ImportError:
            self._tracker_failed=True
        self._next_id=1; self._tracks={}; self._frame=0

    def track(self, frame, conf=.25, device="cpu"):
        if not self._tracker_failed:
            try:
                return self.model.track(source=frame, persist=True, tracker="botsort.yaml", conf=conf,
                                        device=device, verbose=False)
            except Exception:
                # Ultralytics BoT-SORT requires optional lap; remain usable offline.
                self._tracker_failed=True
        result=self.model.predict(source=frame,conf=conf,device=device,verbose=False)[0]
        self._frame += 1
        candidates=[]
        if result.boxes is not None:
            for box in result.boxes:
                vals=box.xyxy[0].tolist(); cx=(vals[0]+vals[2])/2; cy=(vals[1]+vals[3])/2
                candidates.append((box,cx,cy))
        assigned=[]; used=set(); next_tracks={}
        for box,cx,cy in candidates:
            best=None; bestd=100.0
            for tid,(tx,ty,last) in self._tracks.items():
                d=((cx-tx)**2+(cy-ty)**2)**.5
                if tid not in used and self._frame-last<=20 and d<bestd: best,bestd=tid,d
            tid=best if best is not None else self._next_id
            if best is None: self._next_id+=1
            used.add(tid); next_tracks[tid]=(cx,cy,self._frame)
            assigned.append(SimpleNamespace(xyxy=box.xyxy,cls=box.cls,conf=box.conf,id=tid))
        # Keep short-lived lost tracks for re-association.
        self._tracks={**{k:v for k,v in self._tracks.items() if self._frame-v[2]<=20},**next_tracks}
        result.boxes=SimpleNamespace(__iter__=None) if False else _BoxList(assigned)
        return [result]

class _BoxList(list):
    @property
    def id(self): return np.asarray([b.id for b in self],dtype=float) if self else None
