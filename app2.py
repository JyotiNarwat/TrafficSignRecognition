import streamlit as st
from streamlit_webrtc import webrtc_streamer , VideoProcessorBase
import cv2
import av
import tensorflow as tf
import numpy as np
from labels import classes

from ultralytics import YOLO

yolo = YOLO("yolov8n.pt")

IMG_HEIGHT = 30
IMG_WIDTH = 30

model = tf.keras.models.load_model("traffic_sign_model.keras")

class TrafficSignProcessor(VideoProcessorBase):

    def recv(self, frame):

        img = frame.to_ndarray(
            format="bgr24"
        )

        results = yolo(img)

        for result in results:

            for box in result.boxes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                crop = img[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                crop = cv2.resize(
                    crop,
                    (30,30)
                )

                crop = crop.astype(
                    np.float32
                ) / 255.0

                crop = np.expand_dims(
                    crop,
                    axis=0
                )

                prediction = model.predict(
                    crop,
                    verbose=0
                )

                class_id = np.argmax(
                    prediction
                )

                confidence = np.max(
                    prediction
                )

                label = classes.get(
                    class_id,
                    "Unknown"
                )

                cv2.rectangle(
                    img,
                    (x1,y1),
                    (x2,y2),
                    (0,255,0),
                    2
                )

                cv2.putText(
                    img,
                    f"{label} {confidence:.2f}",
                    (x1,y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

    
st.title("🚦 Real-Time Traffic Sign Recognition")

st.write(
    "Click START and Show a traffic sign to the webcam."
)

st.markdown("""
<style>
video {
    width: 900px !important;
    height: auto !important;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

webrtc_streamer(
    key="traffic-sign",
    video_processor_factory=
    TrafficSignProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False,
    }
)
