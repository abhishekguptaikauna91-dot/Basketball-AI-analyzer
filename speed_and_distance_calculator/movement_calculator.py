"""Movement statistics from world/court coordinates only."""
import math

def calculate_tracks(observations,max_speed_m_s=12.0):
    """Rows require track_id,timestamp_s,x_m,y_m; absent calibration => no input rows."""
    grouped={}
    for row in observations: grouped.setdefault(row["track_id"],[]).append(row)
    out=[]
    for tid,rows in grouped.items():
        rows.sort(key=lambda r:r["timestamp_s"]); dist=0.; duration=0.; speeds=[]
        for a,b in zip(rows,rows[1:]):
            dt=b["timestamp_s"]-a["timestamp_s"]
            if dt<=0: continue
            d=math.hypot(b["x_m"]-a["x_m"],b["y_m"]-a["y_m"]); speed=d/dt
            if speed>max_speed_m_s: continue
            dist+=d; duration+=dt; speeds.append(speed)
        out.append({"track_id":tid,"distance_m":round(dist,2),"average_speed_m_s":round(dist/duration,2) if duration else None,"valid_intervals":len(speeds)})
    return out
