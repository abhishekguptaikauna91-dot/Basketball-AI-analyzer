"""Robust, bounded-memory player uniform clustering with explicit uncertainty.

Team names are deterministic but inherently arbitrary unless reference track IDs
or manual assignments are supplied in config.yaml.
"""
from collections import defaultdict, deque

import cv2
import numpy as np


class TeamAssigner:
    def __init__(self, manual_assignments=None, min_samples=5, min_separation=.12,
                 confidence_threshold=.25, max_observations=24,
                 team_a_reference_track_ids=None, team_b_reference_track_ids=None):
        self.manual = {int(k): str(v) for k, v in (manual_assignments or {}).items()}
        self.min_samples = max(1, int(min_samples))
        self.min_separation = float(min_separation)
        self.confidence_threshold = float(confidence_threshold)
        self.max_observations = max(5, int(max_observations))
        self.references = {
            "Team A": {int(x) for x in (team_a_reference_track_ids or [])},
            "Team B": {int(x) for x in (team_b_reference_track_ids or [])},
        }
        if self.references["Team A"] & self.references["Team B"]:
            raise ValueError("A track ID cannot be a reference for both teams")
        self.samples = defaultdict(lambda: deque(maxlen=self.max_observations))
        self.assignments = {}
        self._centers = None
        self._calibration = "pending"
        self._separation = 0.0

    @staticmethod
    def _uniform_feature(frame, xyxy):
        """Return a robust upper-torso LAB feature and quality, or None."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = map(lambda x: int(round(float(x))), xyxy)
        bw, bh = x2 - x1, y2 - y1
        if bw < 12 or bh < 24 or bw / max(1, bh) > 1.5:
            return None
        # Central chest avoids much of the head, arms, shorts and court.
        xa, xb = max(0, x1 + int(.20*bw)), min(w, x2 - int(.20*bw))
        ya, yb = max(0, y1 + int(.20*bh)), min(h, y1 + int(.56*bh))
        crop = frame[ya:yb, xa:xb]
        if crop.shape[0] < 8 or crop.shape[1] < 6:
            return None
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        # Suppress near-black shadows; retain bright/white jerseys. Median LAB
        # statistics prevent a few highlights from dominating the feature.
        valid = hsv[:, :, 2] > 28
        pixels = lab[valid]
        if len(pixels) < max(12, int(crop.shape[0] * crop.shape[1] * .18)):
            return None
        # Median and trimmed spread are resistant to small areas of skin/background.
        med = np.median(pixels, axis=0).astype(np.float32) / 255.0
        q10, q90 = np.percentile(pixels[:, 1:3], [10, 90], axis=0)
        chroma_spread = float(np.mean(q90 - q10) / 255.0)
        # A compact HSV histogram distinguishes jersey hues while LAB medians
        # retain useful signal for low-saturation white uniforms.
        hs = cv2.calcHist([hsv], [0], valid.astype(np.uint8), [12], [0, 180]).flatten().astype(np.float32)
        hs /= hs.sum() + 1e-9
        feature = np.concatenate((med, np.array([chroma_spread], dtype=np.float32), hs * .45))
        area_quality = min(1.0, len(pixels) / 900.0)
        quality = float(area_quality * max(.25, 1.0 - chroma_spread))
        return feature, quality

    def observe(self, track_id, frame, xyxy):
        if track_id is None:
            return
        tid = int(track_id)
        if tid in self.manual:
            return
        measured = self._uniform_feature(frame, xyxy)
        if measured is None:
            return
        feature, quality = measured
        if quality >= .12:
            self.samples[tid].append((feature, quality))

    def _track_features(self):
        ids, features = [], []
        for tid, records in self.samples.items():
            if len(records) < self.min_samples:
                continue
            arr = np.stack([x[0] for x in records])
            weights = np.array([x[1] for x in records], dtype=np.float32)
            features.append(np.average(arr, axis=0, weights=weights))
            ids.append(tid)
        return ids, np.asarray(features, dtype=np.float32)

    def _calibrate(self, ids, x):
        if len(ids) < 2:
            return False
        # Anchors determine which cluster is Team A/B when both are supplied.
        anchor_a = [i for i, tid in enumerate(ids) if tid in self.references["Team A"]]
        anchor_b = [i for i, tid in enumerate(ids) if tid in self.references["Team B"]]
        anchored = bool(anchor_a and anchor_b)
        if anchored:
            centers = np.stack((np.mean(x[anchor_a], axis=0), np.mean(x[anchor_b], axis=0)))
        else:
            # Deterministic farthest-pair initialization, independent of tracker ID order.
            distances = np.linalg.norm(x[:, None] - x[None, :], axis=2)
            i, j = np.unravel_index(np.argmax(distances), distances.shape)
            if distances[i, j] < self.min_separation:
                return False
            centers = np.stack((x[i], x[j]))
        # Reference track colors are trusted anchors: do not let an ambiguous
        # automatically assigned group drag either reference centroid.
        if not anchored:
            for _ in range(50):
                labels = np.argmin(np.linalg.norm(x[:, None] - centers[None, :], axis=2), axis=1)
                if len(set(labels.tolist())) < 2:
                    break
                updated = np.stack([x[labels == k].mean(axis=0) for k in (0, 1)])
                if np.allclose(centers, updated, atol=1e-4):
                    centers = updated
                    break
                centers = updated
        labels = np.argmin(np.linalg.norm(x[:, None] - centers[None, :], axis=2), axis=1)
        if len(set(labels.tolist())) < 2:
            return False
        self._separation = float(np.linalg.norm(centers[0] - centers[1]))
        if self._separation < self.min_separation:
            return False
        # If no reference tracks were given, canonicalize group names by LAB color.
        if not (anchor_a and anchor_b):
            order = sorted((0, 1), key=lambda k: tuple(centers[k][:3]))
            centers = centers[order]
            labels = np.array([order.index(v) for v in labels])
            self._calibration = "automatic_color_order"
        else:
            self._calibration = "reference_tracks"
        self._centers = centers
        for tid, vec in zip(ids, x):
            self._set_result(tid, vec)
        return True

    def _set_result(self, tid, vec):
        if self._centers is None:
            return
        if tid in self.references["Team A"] or tid in self.references["Team B"]:
            team = "Team A" if tid in self.references["Team A"] else "Team B"
            self.assignments[int(tid)] = {"team": team, "confidence": 1.0, "confirmed": True, "source": "reference_track"}
            return
        d = np.linalg.norm(self._centers - vec[None, :], axis=1)
        label = int(np.argmin(d))
        confidence = float((d[1-label] - d[label]) / (d.sum() + 1e-9))
        if confidence >= self.confidence_threshold:
            self.assignments[int(tid)] = {
                "team": f"Team {'A' if label == 0 else 'B'}",
                "confidence": round(confidence, 3), "confirmed": True,
                "source": self._calibration,
            }
        else:
            self.assignments.pop(int(tid), None)

    def finalize(self):
        ids, x = self._track_features()
        if self._centers is None:
            self._calibrate(ids, x)
        else:
            for tid, vec in zip(ids, x):
                self._set_result(tid, vec)
        return self.results()

    def results(self):
        out = {i: dict(v) for i, v in self.assignments.items()}
        for tid, team in self.manual.items():
            out[int(tid)] = {"team": team, "confidence": 1.0, "confirmed": True, "source": "manual"}
        return out

    def diagnostics(self):
        return {
            "calibration": self._calibration,
            "cluster_separation": round(self._separation, 4),
            "tracks_with_reliable_samples": sum(len(v) >= self.min_samples for v in self.samples.values()),
            "confirmed_tracks": len(self.results()),
            "unknown_tracks": sum(len(v) >= self.min_samples for v in self.samples.values()) - len(self.assignments),
            "samples_per_track": {str(k): len(v) for k, v in self.samples.items()},
        }
