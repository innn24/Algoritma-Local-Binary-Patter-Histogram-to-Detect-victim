import cv2
import mediapipe as mp
import numpy as np
import os
import time
import base64
import serial  # ✅ Tambahan untuk komunikasi ke Arduino
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
from sklearn.metrics.pairwise import cosine_similarity

# ====== Estimasi Jarak ======
FOCAL_LENGTH = 615
AVERAGE_HUMAN_HEIGHT_CM = 165

# ====== Inisialisasi Firebase Admin SDK ======
firebase_connected = True
try:
    cred = credentials.Certificate("lbph-47b5a-firebase-adminsdk-fbsvc-85bc3b162f.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://lbph-47b5a-default-rtdb.asia-southeast1.firebasedatabase.app/'
    })
    ref = db.reference('/')
except Exception as e:
    print(f"⚠ Gagal koneksi Firebase: {e}")
    firebase_connected = False

# ====== Inisialisasi Serial ke Arduino ======
try:
    arduino = serial.Serial('COM3', 9600, timeout=1)  # Ganti port jika perlu
    time.sleep(2)
    print("✅ Serial ke Arduino terhubung.") 
except Exception as e:
    print(f"⚠ Gagal konek Arduino: {e}")
    arduino = None

# ====== Fungsi kirim data ke Arduino ======
def send_to_arduino(similarity, distance_cm):
    if arduino:
        try:
            data = f"SIM:{similarity:.1f}|DIST:{distance_cm:.1f}\n"
            arduino.write(data.encode())
            print(f"📤 Terkirim ke Arduino: {data.strip()}")
        except Exception as e:
            print(f"⚠ Gagal kirim ke Arduino: {e}")

# ====== Fungsi Ekstrak LBPH Histogram ======
def extract_lbph(gray_img):
    lbph = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    lbph.train([gray_img], np.array([0]))
    hist = lbph.getHistograms()[0]
    return hist.flatten()

# ====== Fungsi Konversi Gambar ke Base64 ======
def image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    img_bytes = buffer.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

# ====== Load Sample LBPH dari Folder ======
sample_folder = "samples"
sample_hists = []

if os.path.exists(sample_folder):
    for filename in os.listdir(sample_folder):
        if filename.lower().endswith((".jpg", ".png")):
            img_path = os.path.join(sample_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (128, 128))
            hist = extract_lbph(img)
            sample_hists.append(hist)
    print(f"✅ {len(sample_hists)} sampel LBPH dimuat.")
else:
    print("❌ Folder 'samples/' tidak ditemukan.")
    exit()

# ====== Inisialisasi MediaPipe ======
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# ====== Folder penyimpanan lokal ======
detected_folder = "detected"
if not os.path.exists(detected_folder):
    os.makedirs(detected_folder)

cap = cv2.VideoCapture(0)
last_detection_state = -1
detected_counter = 0
required_detection_frames = 5

print("🎥 Menjalankan deteksi real-time... Tekan ESC untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    detection_value = 0
    similarity_display_text = "Kemiripan Sample: -"
    label = ""
    distance_text = "Jarak: -"
    distance_cm = 0.0

    if results.pose_landmarks:
        xs = [lm.x * w for lm in results.pose_landmarks.landmark]
        ys = [lm.y * h for lm in results.pose_landmarks.landmark]
        x_min, y_min = int(min(xs)), int(min(ys))
        x_max, y_max = int(max(xs)), int(max(ys))

        margin_top = 40
        x_min, y_min = max(0, x_min), max(0, y_min - margin_top)
        x_max, y_max = min(w, x_max), min(h, y_max)

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        crop = frame[y_min:y_max, x_min:x_max]

        bbox_height = y_max - y_min
        if bbox_height > 0:
            distance_cm = (FOCAL_LENGTH * AVERAGE_HUMAN_HEIGHT_CM) / bbox_height
            distance_text = f"Jarak: {distance_cm:.1f} cm"

        if crop.size != 0:
            gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            resized_crop = cv2.resize(gray_crop, (128, 128))
            detected_hist = extract_lbph(resized_crop)

            similarities = [cosine_similarity([ref], [detected_hist])[0][0] for ref in sample_hists]
            max_similarity = max(similarities)
            similarity = max_similarity * 100
            similarity_text = f"{similarity:.1f}%"
            similarity_display_text = f"{similarity_text}"
            label = "Manusia"
            detection_value = 1
            detected_counter += 1
        else:
            detected_counter = 0
    else:
        detected_counter = 0

    if detection_value == 1:
        label_text = f"{label} | Kemiripan Sample: {similarity_display_text}"
        cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(frame, distance_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    if detected_counter == required_detection_frames and last_detection_state != 1:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        if firebase_connected:
            try:
                img_b64 = image_to_base64(frame)
                history_ref = ref.child("history").push()
                history_ref.set({
                    "timestamp": now,
                    "image": img_b64,
                    "similarity": similarity_text,
                    "distance": distance_text
                })
                ref.update({"detection": 1})
                print("📡 Data terkirim ke Firebase.")
            except Exception as e:
                print(f"⚠ Gagal kirim ke Firebase: {e}")
                filename = f"{detected_folder}/detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Tersimpan lokal: {filename}")

        send_to_arduino(similarity, distance_cm)  # ✅ Kirim ke Arduino

        last_detection_state = 1
        detected_counter = 0

    if detection_value == 0 and last_detection_state != 0:
        if firebase_connected:
            try:
                ref.update({"detection": 0})
            except Exception as e:
                print(f"⚠ Gagal update status ke Firebase: {e}")
        last_detection_state = 0

    cv2.imshow("Realtime Multiple Sample LBPH + MediaPipe", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
if arduino:
    arduino.close()