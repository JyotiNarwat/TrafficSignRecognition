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
    "Click START and Show a traffic sign image(using phone or printout) to the webcam."
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

st.text("""
    Model was trained on the following set of signs:

    0: Speed limit (20km/h)
    1: Speed limit (30km/h)
    2: Speed limit (50km/h)
    3: Speed limit (60km/h)
    4: Speed limit (70km/h)
    5: Speed limit (80km/h)
    6: End of speed limit (80km/h)
    7: Speed limit (100km/h)
    8: Speed limit (120km/h)
    9: No passing
    10: No passing vehicles over 3.5 tons
    11: Right-of-way at intersection
    12: Priority road
    13: Yield
    14: Stop
    15: No vehicles
    16: Vehicles over 3.5 tons prohibited
    17: No entry
    18: General caution
    19: Dangerous curve left
    20: Dangerous curve right
    21: Double curve
    22: Bumpy road
    23: Slippery road
    24: Road narrows on the right
    25: Road work
    26: Traffic signals
    27: Pedestrians
    28: Children crossing
    29: Bicycles crossing
    30: Beware of ice/snow
    31: Wild animals crossing
    32: End speed and passing limits
    33: Turn right ahead
    34: Turn left ahead
    35: Ahead only
    36: Go straight or right
    37: Go straight or left
    38: Keep right
    39: Keep left
    40: Roundabout mandatory
    41: End of no passing
    42: End no passing vehicles over 3.5 tons

    you may test the model using any of these signs
""")
