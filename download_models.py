from huggingface_hub import snapshot_download
import os

# Tạo thư mục checkpoints nếu chưa có (dù máy bạn báo có rồi, cứ để lệnh này cho chắc)
os.makedirs("checkpoints", exist_ok=True)

print("⬇️ 1. Đang tải Model StereoGen (Sinh ảnh trái/phải)...")
# Tải StereoGen từ repo gốc
snapshot_download(repo_id="Windsrain/ZeroStereo", allow_patterns=["StereoGen/*"], local_dir="checkpoints")

print("⬇️ 2. Đang tải Model Zero-IGEV (Tính độ sâu/Disparity)...")
# Tải Zero-IGEV để đánh giá hoặc sinh map độ sâu
snapshot_download(repo_id="Windsrain/ZeroStereo", allow_patterns=["Zero-IGEV-Stereo/*"], local_dir="checkpoints")

print("✅ Đã tải xong! Bạn có thể chạy code được rồi.")