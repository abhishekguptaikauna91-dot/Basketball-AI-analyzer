"""Court pose inference; keypoint semantics are intentionally not assumed."""
from ultralytics import YOLO

class CourtKeypointDetector:
    def __init__(self, model_path):
        self.model=YOLO(str(model_path))
        self.names=self.model.names
        if self.model.task != "pose":
            raise ValueError(f"Court checkpoint must be pose task, got {self.model.task}")

    def predict(self, frame, conf=.2, device="cpu"):
        return self.model.predict(source=frame, conf=conf, device=device, verbose=False)
