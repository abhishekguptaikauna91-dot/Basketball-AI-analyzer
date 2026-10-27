import cv2

def open_video(path):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): raise RuntimeError(f"Cannot open input video: {path}")
    fps=float(cap.get(cv2.CAP_PROP_FPS)); width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps<=0 or width<=0 or height<=0:
        cap.release(); raise ValueError(f"Invalid video metadata (fps={fps}, width={width}, height={height})")
    return cap,{"fps":fps,"width":width,"height":height,"frame_count":count}

def make_writer(path,fps,width,height):
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(width,height))
    if not writer.isOpened(): raise RuntimeError(f"Cannot create output video: {path}")
    return writer
