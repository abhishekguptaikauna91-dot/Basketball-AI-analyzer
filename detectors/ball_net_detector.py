
import cv2
from ultralytics import YOLO


class BallNetDetector:
    def __init__(self, model_path):
        self.model = YOLO(str(model_path))
        self.names = self.model.names
        print("Model loaded successfully!")
        print("Classes:", self.names)

    def class_id(self, wanted):
        key = wanted.lower().replace(" ", "_").replace("-", "_")
        for idx, name in self.names.items():
            if str(name).lower().replace(" ", "_").replace("-", "_") == key:
                return int(idx)
        raise ValueError(f"Class {wanted!r} missing from ball/net model: {self.names}")

    def predict(self, frame, conf=.25, device="cpu"):
        return self.model.predict(source=frame, conf=conf, device=device, verbose=False)

    def detect_video(self, video_path, output_path):
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            fps = 30

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot create output video: {output_path}")

        # Select only the three required classes
        required = {"ball", "net", "ball_in_basket"}

        def normalize(name):
            return name.lower().replace(" ", "_").replace("-", "_")

        class_ids = [
            i for i, name in self.model.names.items()
            if normalize(name) in required
        ]

        if not class_ids:
            cap.release()
            writer.release()
            raise ValueError(
                f"Required classes not found. Model classes: {self.model.names}"
            )

        print("Detecting:", [self.model.names[i] for i in class_ids])

        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                results = self.model.predict(
                    source=frame,
                    conf=0.25,
                    classes=class_ids,
                    verbose=False
                )

                # Draw bounding boxes, class names, and confidence scores
                frame = results[0].plot()
                writer.write(frame)

                frame_count += 1

                if frame_count % 100 == 0:
                    print(f"Processed {frame_count} frames")

        finally:
            cap.release()
            writer.release()

        print(f"Finished processing {frame_count} frames.")
        print(f"Output saved to: {output_path}")
