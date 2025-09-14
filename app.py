import cv2
import numpy as np
import mediapipe as mp
import joblib
import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
from streamlit_autorefresh import st_autorefresh
# === CONFIG ===
MODEL_PATH = "model(1).pkl"
CLASSES_TXT = "classes(1).txt"

# Landmark subset (sama dengan training)
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

# === Load Model & Classes ===
model = joblib.load(MODEL_PATH)
with open(CLASSES_TXT, "r") as f:
    classes = [l.strip() for l in f]

# === Penjelasan Ekspresi ===
EXPLANATIONS = {
    "datar": "Ekspresi datar menandakan kondisi netral atau tanpa emosi dominan.",
    "kaget": "Ekspresi kaget muncul saat menghadapi sesuatu yang tiba-tiba atau tak terduga.",
    "marah": "Ekspresi marah biasanya terkait dengan perasaan terganggu, frustrasi, atau tidak setuju.",
    "sedih": "Ekspresi sedih mengindikasikan perasaan kehilangan, kecewa, atau terluka.",
    "senang": "Ekspresi senang menandakan kebahagiaan, kepuasan, dan suasana hati positif."
}

# === MediaPipe FaceMesh ===
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# === Streamlit UI ===
st.set_page_config(page_title="Realtime Emotion Recognition", layout="wide")

st.title("😊 Realtime Emotion Recognition untuk Bimbingan Konseling")
st.markdown(
    """
    Sistem ini mendeteksi **ekspresi wajah** secara realtime melalui webcam.
    <br>Deteksi ekspresi yang tersedia: **Datar, Kaget, Marah, Sedih, Senang**.
    <br><br>
    **Tujuan:** Membantu konselor memahami kondisi emosional siswa/klien secara objektif.
    """,
    unsafe_allow_html=True
)

# === Video Processor ===
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.label = "Menunggu wajah..."

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            coords = np.array([[lm[i].x, lm[i].y] for i in IMPORTANT_LANDMARKS], dtype=np.float32).flatten()

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

                pred = model.predict([normalized])[0]
                self.label = classes[pred]

                cv2.putText(img, f"Ekspresi: {self.label}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

                # Gambar landmark
                h_img, w_img, _ = img.shape
                for i in IMPORTANT_LANDMARKS:
                    x = int(lm[i].x * w_img)
                    y = int(lm[i].y * h_img)
                    cv2.circle(img, (x, y), 1, (0, 255, 255), -1)
        else:
            self.label = "Wajah tidak terdeteksi"

        # Simpan ke session_state agar bisa dibaca sidebar
        st.session_state["current_label"] = self.label

        return av.VideoFrame.from_ndarray(img, format="bgr24")


RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:relay1.expressturn.com:3478"],
            "username": "efree",
            "credential": "efreepass"
        }
    ]
}
ctx = webrtc_streamer(
    key="emotion",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,  
    video_processor_factory=EmotionProcessor,
    media_stream_constraints={"video": True, "audio": False},
)
# === Sidebar Info (auto-refresh + penjelasan) ===
st.sidebar.header("📘 Penjelasan Ekspresi")
st_autorefresh(interval=900, key="refresh_sidebar")

if ctx and ctx.video_processor:
    raw_label = ctx.video_processor.label
    st.sidebar.subheader(f"Ekspresi Terdeteksi: **{raw_label}**")

    # Normalisasi label
    norm_label = raw_label.lower().replace("ekspresi", "").strip()

    if norm_label in EXPLANATIONS:
        st.sidebar.write(EXPLANATIONS[norm_label])
    else:
        st.sidebar.write("Ekspresi tidak dikenali.")
else:
    st.sidebar.write("Menunggu wajah...")

