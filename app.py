# app.py
import cv2
import numpy as np
import mediapipe as mp
import joblib
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# === Load model & classes ===
MODEL_PATH = "model(1).pkl"
CLASSES_TXT = "classes(1).txt"

model = joblib.load(MODEL_PATH)
with open(CLASSES_TXT, "r") as f:
    classes = [l.strip() for l in f]

# Landmark subset (harus sama dengan training)
RIGHT_EYE = list(range(33, 133))
LEFT_EYE  = list(range(362, 463))
IRIS_RIGHT = [468, 469, 470, 471]
IRIS_LEFT  = [472, 473, 474, 475]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 193, 122, 46, 53]
LEFT_EYEBROW  = [336, 296, 334, 293, 300, 276, 283, 352, 276, 282]
MOUTH_OUTER = list(range(61, 81)) + list(range(308, 329))
MOUTH_INNER = list(range(81, 91)) + list(range(311, 321))
NOSE = [1, 2, 98, 327, 97, 326, 168, 195, 5, 4, 19, 94, 141, 208, 49, 279]
JAWLINE = [127, 234, 93, 132, 58, 172, 150, 176, 148, 152, 377, 400, 378, 379, 365, 397]

IMPORTANT_LANDMARKS = list(set(
    RIGHT_EYE + LEFT_EYE + IRIS_RIGHT + IRIS_LEFT +
    RIGHT_EYEBROW + LEFT_EYEBROW +
    MOUTH_OUTER + MOUTH_INNER +
    NOSE + JAWLINE
))

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# === Video Transformer untuk Streamlit-WebRTC ===
class FaceLandmarkTransformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> np.ndarray:
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            coords = np.array([[lm[i].x, lm[i].y] for i in IMPORTANT_LANDMARKS],
                              dtype=np.float32).flatten()

            xs = coords[0::2]; ys = coords[1::2]
            minx, maxx = xs.min(), xs.max()
            miny, maxy = ys.min(), ys.max()
            w, h = maxx - minx, maxy - miny

            if w > 0 and h > 0:
                xs_n = (xs - minx) / w
                ys_n = (ys - miny) / h
                normalized = np.empty_like(coords)
                normalized[0::2] = xs_n
                normalized[1::2] = ys_n

                # Prediksi kelas
                pred = model.predict([normalized])[0]
                label = classes[pred]

                # Tampilkan label di atas frame
                cv2.putText(img, label, (30, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (0, 255, 0), 3)

                # Gambar landmark subset
                h_img, w_img, _ = img.shape
                for i in IMPORTANT_LANDMARKS:
                    x = int(lm[i].x * w_img)
                    y = int(lm[i].y * h_img)
                    cv2.circle(img, (x, y), 1, (0, 255, 255), -1)

        return img


# === Streamlit UI ===
import streamlit as st
st.title("📷 Realtime Face Landmark Classification")
st.write("Webcam streaming dengan prediksi kelas wajah realtime.")

webrtc_streamer(
    key="face-landmark",
    video_transformer_factory=FaceLandmarkTransformer,
    media_stream_constraints={"video": True, "audio": False}
)
