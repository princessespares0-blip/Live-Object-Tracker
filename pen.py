import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import time
from collections import Counter


@st.cache_resource
def load_model():

    return YOLO("yolov8s.pt")

model = load_model()

st.set_page_config(page_title="Live Detection", layout="wide")

st.title("Live Object Detection & Tracking")

confidence = st.sidebar.slider("Confidence", 0.25, 1.0, 0.5)

state = {"prev_time": time.time()}

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    results = model.predict(img, conf=confidence, verbose=False)
    result = results[0]

    annotated = img.copy()

    counts = Counter()

    if result.boxes is not None:
        names = result.names

        for box in result.boxes:
            conf = float(box.conf[0])
            if conf < confidence:
                continue

            cls = int(box.cls[0])
            label = names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            counts[label] += 1

            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                annotated,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    
    curr = time.time()
    fps = 1 / max(curr - state["prev_time"], 1e-6)
    state["prev_time"] = curr

    cv2.putText(
        annotated,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    
    for obj in ["person", "cell phone", "bottle"]:
        if obj in counts:
            cv2.putText(
                annotated,
                f"ALERT: {obj.upper()}",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    
    y = 110
    for obj, num in counts.items():
        cv2.putText(
            annotated,
            f"{obj}: {num}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        y += 25

    return av.VideoFrame.from_ndarray(annotated, format="bgr24")

webrtc_streamer(
    key="fixed-app",
    video_frame_callback=video_frame_callback,
    async_processing=True,
    media_stream_constraints={"video": True, "audio": False},
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
)