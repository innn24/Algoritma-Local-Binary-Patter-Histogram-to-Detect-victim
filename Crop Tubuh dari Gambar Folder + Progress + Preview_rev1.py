import cv2
import mediapipe as mp
import os
import time

# Folder input dan output
input_folder = "screenshots"         # Ganti dengan folder kamu
output_folder = "samples"            # Folder hasil crop

# Buat folder output jika belum ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Inisialisasi MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Ambil semua file gambar
image_files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".png"))]
total_files = len(image_files)
counter = 0

# Margin tambahan ke atas (untuk memastikan kepala ikut tercrop)
margin_top = 40

print(f"🔍 Memproses {total_files} file dari folder '{input_folder}'...\n")

for i, filename in enumerate(image_files):
    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)
    if image is None:
        print(f"⚠️  Gagal membuka {filename}, dilewati.")
        continue

    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    progress_bar = f"[{'=' * int((i+1)/total_files*30):30}] {i+1}/{total_files}"
    print(f"\r{progress_bar}", end="")

    if results.pose_landmarks:
        xs = [lm.x * w for lm in results.pose_landmarks.landmark]
        ys = [lm.y * h for lm in results.pose_landmarks.landmark]
        x_min, y_min = int(min(xs)), int(min(ys))
        x_max, y_max = int(max(xs)), int(max(ys))

        x_min = max(0, x_min)
        y_min = max(0, y_min - margin_top)  # Naikkan crop ke atas
        x_max = min(w, x_max)
        y_max = min(h, y_max)

        body_crop = image[y_min:y_max, x_min:x_max]
        if body_crop.size != 0:
            gray = cv2.cvtColor(body_crop, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (128, 128))
            save_path = os.path.join(output_folder, f"sample_{counter}.jpg")
            cv2.imwrite(save_path, resized)
            counter += 1

            # Tampilkan gambar sementara
            preview = cv2.resize(image, (640, 480))
            cv2.rectangle(preview, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.imshow("Preview Proses", preview)
            cv2.waitKey(1)  # tampilkan cepat

cv2.destroyAllWindows()
print(f"\n✅ Selesai! Total sample tubuh yang disimpan: {counter}")