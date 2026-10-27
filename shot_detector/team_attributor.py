"""Conservative team-level made-basket attribution from recent observed ball contact."""
import math

class ScoringTeamAttributor:
    def __init__(self,window_seconds=1.8,max_distance_px=115,min_support_frames=3,min_team_confidence=.18):
        self.window_seconds=window_seconds
        self.max_distance_px=max_distance_px
        self.min_support_frames=min_support_frames
        self.min_team_confidence=min_team_confidence

    def attribute(self,event,ball_history,player_history,team_assignments):
        """Return a probable scoring team only with repeated, unambiguous ball proximity.

        The associated track is evidence of likely team possession, not proof that the
        player took the shot. Individual shooter attribution is deliberately omitted.
        """
        event_time=float(event.get("timestamp_s",0)); event_frame=event.get("frame")
        earliest=event_time-self.window_seconds
        rows_by_frame={}
        for rows in player_history:
            if not rows: continue
            for row in rows:
                rows_by_frame.setdefault(int(row["frame"]),[]).append(row)
        support={}; observations=0
        for ball in ball_history:
            if not ball.get("observed"): continue
            ts=float(ball.get("timestamp_s",0)); frame=int(ball.get("frame",-1))
            if not (earliest<=ts<=event_time) or (event_frame is not None and frame>int(event_frame)): continue
            candidates=[]
            for player in rows_by_frame.get(frame,[]):
                tid=player.get("track_id")
                if tid in (None,""): continue
                center_y=player.get("center_y_px",player["foot_y_px"])
                d=math.hypot(float(ball["x_px"])-float(player["center_x_px"]),float(ball["y_px"])-float(center_y))
                candidates.append((d,int(tid)))
            candidates.sort()
            if not candidates or candidates[0][0]>self.max_distance_px: continue
            if len(candidates)>1 and candidates[1][0]-candidates[0][0]<18: continue
            tid=candidates[0][1]; assignment=team_assignments.get(tid,{})
            team=assignment.get("team"); confidence=float(assignment.get("confidence",0) or 0)
            if team not in ("Team A","Team B") or not assignment.get("confirmed") or confidence<self.min_team_confidence: continue
            support.setdefault((tid,team),set()).add(frame); observations+=1
        if not support:
            return {"scoring_team":"","team_attribution_confidence":0.0,"associated_track_id":"","team_attribution_evidence":"insufficient recent unambiguous ball/player evidence"}
        ranked=sorted(((len(frames),tid,team) for (tid,team),frames in support.items()),reverse=True)
        count,tid,team=ranked[0]; total=sum(n for n,_,_ in ranked)
        confidence=count/max(1,total)
        if count<self.min_support_frames or confidence<.70:
            return {"scoring_team":"","team_attribution_confidence":round(confidence,3),"associated_track_id":"","team_attribution_evidence":f"ambiguous ball/player proximity ({count} supporting frames)"}
        return {"scoring_team":team,"team_attribution_confidence":round(confidence,3),"associated_track_id":tid,"team_attribution_evidence":f"{count} distinct observed-ball frames near track {tid}; team inferred from uniform cluster"}
