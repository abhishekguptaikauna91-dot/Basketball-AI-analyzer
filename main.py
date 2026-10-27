"""Streaming basketball video analysis using the project's trained YOLO models."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import time
import cv2
from collections import deque
from pathlib import Path

from detectors.ball_net_detector import BallNetDetector
from detectors.player_detector import PlayerDetector
from detectors.court_keypoint_detector import CourtKeypointDetector
from utils.video_utils import open_video, make_writer
from utils.player_heatmap import create_player_heatmap
from shot_detector.shot_detector import ShotDetector
from ball_acquisition.ball_tracker import BallTracker
from team_assigner.team_assigner import TeamAssigner
from pass_and_interception_detector.event_detector import InteractionDetector
from speed_and_distance_calculator.movement_calculator import calculate_tracks
from drawers.video_drawer import VideoRenderer
from shot_detector.team_attributor import ScoringTeamAttributor
import yaml

BASE_DIR = Path(__file__).resolve().parent
LOG = logging.getLogger("basketball")


def _write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(input_path: Path, output_dir: Path, device: str = "auto", conf: float | None = None,
            max_frames: int | None = None, team_colors: bool = True, config_path: Path | None = None):
    config_path = config_path or BASE_DIR / "configs/config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    conf = float(conf if conf is not None else config.get("inference", {}).get("confidence", .25))
    configured_models = config.get("models", {})
    model_paths = {
        name: (BASE_DIR / configured_models.get(name, default)).resolve()
        for name, default in {"ball_net":"models/ball_net/best.pt","player":"models/player/best.pt","court":"models/court_keypoints/best.pt"}.items()
    }
    for label, path in [("input video", input_path), *model_paths.items()]:
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    # Resolve and validate all models once, before opening the output writer.
    LOG.info("Loading ball/net model: %s", model_paths["ball_net"])
    ball = BallNetDetector(model_paths["ball_net"])
    LOG.info("Loading player model: %s", model_paths["player"])
    players = PlayerDetector(model_paths["player"])
    LOG.info("Loading court keypoint model: %s (large checkpoint; this may take a while)", model_paths["court"])
    court = CourtKeypointDetector(model_paths["court"])
    chosen_device = "0" if device == "auto" and __import__("torch").cuda.is_available() else ("cpu" if device == "auto" else device)
    LOG.info("Models ready; processing on %s", chosen_device)
    cap, meta = open_video(input_path)
    fps, width, height = meta["fps"], meta["width"], meta["height"]
    visual_config = config.get("visual_config", "configs/visual_config.yaml")
    renderer = VideoRenderer(BASE_DIR / visual_config, fps=fps)
    output_video = output_dir / f"{input_path.stem}_analyzed.mp4"
    writer = make_writer(output_video, fps, width, height)
    shot_cfg=config.get("scoring", {})
    basket_events = ShotDetector(threshold=shot_cfg.get("confidence",.45),confirm_frames=shot_cfg.get("confirmation_frames",2),gap_tolerance_s=shot_cfg.get("gap_tolerance_seconds",.5),cooldown_s=shot_cfg.get("cooldown_seconds",2.0))
    ball_cfg=config.get("ball", {})
    ball_tracker = BallTracker(max_jump_px=ball_cfg.get("max_jump_px",180),max_gap_frames=ball_cfg.get("max_gap_frames",5))
    team_cfg=config.get("team_assignment", {})
    team_assigner = TeamAssigner(
        manual_assignments=team_cfg.get("manual_assignments", {}),
        min_samples=team_cfg.get("min_observations", 5),
        min_separation=team_cfg.get("min_separation", .12),
        confidence_threshold=team_cfg.get("confidence_threshold", .25),
        max_observations=team_cfg.get("max_observations_per_track", 24),
        team_a_reference_track_ids=team_cfg.get("team_a_reference_track_ids", []),
        team_b_reference_track_ids=team_cfg.get("team_b_reference_track_ids", []),
    )
    attributor = ScoringTeamAttributor(window_seconds=config.get("scoring",{}).get("attribution_window_seconds",1.8),max_distance_px=config.get("scoring",{}).get("attribution_max_distance_px",115),min_support_frames=config.get("scoring",{}).get("attribution_min_support_frames",3))
    interactions = InteractionDetector()
    player_rows, ball_rows, score_rows, pass_rows = [], [], [], []
    player_history=deque(maxlen=max(20,int(fps*2.5)))
    ball_history=deque(maxlen=max(20,int(fps*2.5)))
    team_makes={"Team A":0,"Team B":0}; unassigned_makes=0; confirmed_made_count=0
    frame_no = 0
    start = time.time()
    progress_interval = max(1, int(fps * (0.2 if chosen_device == "cpu" else 1.0)))
    ball_class = ball.class_id("ball")
    basket_class = ball.class_id("ball_in_basket")
    net_class = ball.class_id("net")
    try:
        while max_frames is None or frame_no < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            ts = frame_no / fps
            bres = ball.predict(frame, conf=conf, device=chosen_device)
            pres = players.track(frame, conf=conf, device=chosen_device)
            kres = court.predict(frame, conf=max(.15, conf * .7), device=chosen_device)
            ball_dets, hoop_dets, basket_visual_dets = [], [], []
            if bres and bres[0].boxes is not None:
                for box in bres[0].boxes:
                    cid = int(box.cls.item()); xyxy = [float(x) for x in box.xyxy[0].tolist()]
                    score = float(box.conf.item())
                    if cid == ball_class:
                        ball_dets.append((xyxy, score))
                    elif cid == basket_class:
                        hoop_dets.append((xyxy, score))
                        basket_visual_dets.append((xyxy,score))
                    elif cid == net_class:
                        basket_visual_dets.append((xyxy,score))
            ev = basket_events.update_frame(frame_no, ts, hoop_dets, bool(ball_dets))
            if ev:
                ev.update(attributor.attribute(ev,ball_history,player_history,team_assigner.results()))
                score_rows.append(ev)
                if ev.get("status")=="confirmed_candidate":
                    confirmed_made_count+=1
                    if ev.get("scoring_team") in team_makes: team_makes[ev["scoring_team"]]+=1
                    else: unassigned_makes+=1
            tracked_ball=ball_tracker.update(frame_no,ts,ball_dets)
            if tracked_ball:
                ball_rows.append(tracked_ball)
            frame_players=[]
            if pres and pres[0].boxes is not None:
                boxes=pres[0].boxes
                ids=boxes.id.tolist() if boxes.id is not None else [None]*len(boxes)
                for box, track_id in zip(boxes,ids):
                    xyxy=[float(x) for x in box.xyxy[0].tolist()]; x1,y1,x2,y2=map(int,xyxy)
                    cx=(x1+x2)//2; foot_y=y2
                    tid=int(track_id) if track_id is not None else ""
                    if tid != "": team_assigner.observe(tid,frame,xyxy)
                    row={"frame":frame_no,"timestamp_s":round(ts,3),"track_id":tid,"x1":x1,"y1":y1,"x2":x2,"y2":y2,"center_x_px":cx,"center_y_px":(y1+y2)//2,"foot_y_px":foot_y,"team":"unassigned","team_confidence":"","team_confirmed":False}
                    player_rows.append(row); frame_players.append(row)
            ie=interactions.update(frame_no,ts,tracked_ball,frame_players)
            if ie: pass_rows.append(ie)
            player_history.append(frame_players)
            if tracked_ball: ball_history.append(tracked_ball)
            if frame_no % 10 == 0: team_assigner.finalize()
            assignments=team_assigner.results()
            # Persist exactly the state shown in this frame. Earlier frames stay
            # unknown if calibration had not produced a confirmed result yet.
            for row in frame_players:
                team_state = assignments.get(row["track_id"])
                if team_state and team_state.get("confirmed"):
                    row["team"] = team_state["team"]
                    row["team_confidence"] = team_state["confidence"]
                    row["team_confirmed"] = True
            frame=renderer.render(frame,players=frame_players,ball_detections=ball_dets,
                ball_track=tracked_ball,baskets=basket_visual_dets,confirmed_event=ev,
                team_assignments=assignments,timestamp_s=ts,frame_no=frame_no,
                court_calibrated=False,team_makes=team_makes,unassigned_makes=unassigned_makes,interaction_event=ie)
            writer.write(frame)
            frame_no += 1
            if frame_no % progress_interval == 0:
                total=meta["frame_count"]
                progress=f"{frame_no}/{total} frames ({100*frame_no/total:.1f}%)" if total>0 else f"{frame_no} frames"
                LOG.info("Processed %s | video %.1f s | elapsed %.1f s",progress,ts,time.time()-start)
    finally:
        cap.release(); writer.release()
    team_results=team_assigner.finalize()
    final_event=basket_events.flush()
    if final_event:
        if final_event.get("frame") is None: final_event["frame"]=max(0,frame_no-1)
        final_event["evidence"]="ball_in_basket detection; end-of-video flush"
        final_event.update(attributor.attribute(final_event,ball_history,player_history,team_results))
        score_rows.append(final_event)
        if final_event.get("status")=="confirmed_candidate":
            confirmed_made_count+=1
            if final_event.get("scoring_team") in team_makes: team_makes[final_event["scoring_team"]]+=1
            else: unassigned_makes+=1
    for row in player_rows:
        row["team_status"] = "confirmed" if row["team_confirmed"] else "unknown"
    _write_csv(output_dir/"player_tracking.csv",player_rows,["frame","timestamp_s","track_id","x1","y1","x2","y2","center_x_px","center_y_px","foot_y_px","team","team_confidence","team_confirmed","team_status"])
    _write_csv(output_dir/"ball_tracking.csv",ball_rows,["frame","timestamp_s","track_id","x_px","y_px","confidence","observed"])
    _write_csv(output_dir/"scoring_events.csv",score_rows,["frame","timestamp_s","confidence","status","evidence","scoring_team","team_attribution_confidence","team_attribution_evidence","associated_track_id","shooter_track_id"])
    _write_csv(output_dir/"pass_and_interception_events.csv",pass_rows,["frame","timestamp_s","event","from_track_id","to_track_id","confidence","status","basis"])
    movement_rows=calculate_tracks([])  # No calibrated court coordinates are currently validated.
    _write_csv(output_dir/"movement_statistics.csv",movement_rows,["track_id","distance_m","average_speed_m_s","valid_intervals"])
    background_cap = cv2.VideoCapture(str(input_path))
    background_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no // 2))
    bg_ok, heatmap_background = background_cap.read()
    if not bg_ok:
        background_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        bg_ok, heatmap_background = background_cap.read()
    background_cap.release()
    heatmap_info = create_player_heatmap(
        player_rows, heatmap_background if bg_ok else None, output_dir / "player_heatmap.png"
    )
    artifacts={p.name:str(p) for p in output_dir.iterdir() if p.is_file()}
    artifacts["match_summary.json"]=str(output_dir/"match_summary.json")
    team_diagnostics = team_assigner.diagnostics()
    confirmed_by_team = {name: sum(v["team"] == name for v in team_results.values()) for name in ("Team A", "Team B")}
    team_diagnostics["confirmed_by_team"] = confirmed_by_team
    if team_diagnostics["calibration"] == "pending":
        LOG.warning("Team colors could not be separated with current reliable samples; player labels remain unknown")
    else:
        LOG.info("Team assignment: %s; confirmed A=%d B=%d unknown=%d; calibration=%s",
                 team_diagnostics["tracks_with_reliable_samples"], confirmed_by_team["Team A"],
                 confirmed_by_team["Team B"], team_diagnostics["unknown_tracks"], team_diagnostics["calibration"])
    summary={"input":str(input_path),"output_video":str(output_video),"artifacts":artifacts,"frames_processed":frame_no,"fps":fps,"duration_processed_s":round(frame_no/fps,3),"elapsed_s":round(time.time()-start,2),"device":chosen_device,"configuration":str(config_path),"models":{k:{"path":str(v),"classes":(ball.names if k=="ball_net" else players.names if k=="player" else court.names)} for k,v in model_paths.items()},"scoring_events":len(score_rows),"confirmed_made_basket_events":confirmed_made_count,"team_made_baskets":team_makes,"unassigned_made_baskets":unassigned_makes,"points_by_team":None,"points_unavailable_reason":"court calibration and 2-point/3-point classification are not validated","scoring_event_statuses":{s:sum(e["status"]==s for e in score_rows) for s in sorted({e["status"] for e in score_rows})},"shot_attempts":None,"missed_shots":None,"shooter_attribution":"individual shooter remains unassigned; scoring team is only a conservative team-possession estimate","team_assignment":{**team_diagnostics,"method":"bounded track-aggregated torso LAB appearance; labels are canonical by color unless reference IDs anchor them"},"player_heatmap":heatmap_info,"court_calibration":"unavailable; pose point semantics and image-to-court mapping remain unverified","distance_speed_heatmaps":"distance and speed heatmaps unavailable without validated court calibration; player_heatmap is an image-space activity map","interaction_events":len(pass_rows),"interaction_event_limit":"low-confidence proximity-derived possession candidates only; not pass/interception proof","warnings":["BoT-SORT unavailable without lap; centroid fallback IDs are temporary","team made-basket attribution requires repeated, unambiguous observed-ball proximity and confirmed team assignment","2-point/3-point values are not displayed without validated court calibration","court keypoint semantics and ordering are unverified; pose points are not drawn and no court mapping is used","ball predictions are marked observed=false and limited to short gaps"]}
    (output_dir/"match_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    LOG.info("Finished %d frames in %.1fs. Output: %s",frame_no,time.time()-start,output_video)
    return summary


def main():
    parser=argparse.ArgumentParser(description="Analyze a basketball video with trained project models")
    parser.add_argument("--input",type=Path,default=BASE_DIR/"input_videos/basketball_match.mp4")
    parser.add_argument("--output-dir",type=Path,default=BASE_DIR/"output_videos/match_run")
    parser.add_argument("--device",default="auto",help="auto, cpu, 0, cuda:0, etc.")
    parser.add_argument("--conf",type=float,default=None)
    parser.add_argument("--max-frames",type=int,default=None,help="Limit frames for a smoke run")
    parser.add_argument("--no-team-colors",action="store_true")
    parser.add_argument("--config",type=Path,default=BASE_DIR/"configs/config.yaml")
    args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
    analyze(args.input.resolve(),args.output_dir.resolve(),args.device,args.conf,args.max_frames,not args.no_team_colors,args.config.resolve())

if __name__=="__main__": main()
