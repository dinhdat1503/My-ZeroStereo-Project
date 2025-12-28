import pymongo
import gridfs
import os
import time

# --- CẤU HÌNH KẾT NỐI MONGODB ---
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["BigDataStereo"]
fs = gridfs.GridFS(db)
collection = db.fs.files

# --- CẤU HÌNH ĐƯỜNG DẪN ---
TEMP_INPUT = "assets/mongo_input.jpg"   # Nơi lưu tạm ảnh tải về
TEMP_OUTPUT_DIR = "mongo_output"        # Nơi chứa kết quả AI
TEMP_RESULT = "mongo_output/mongo_input.png" # Tên file kết quả mong đợi

def process_one_image():
    # 1. Tìm một ảnh trong DB có trạng thái là 'waiting'
    task = collection.find_one({"status": "waiting"})
    
    if task:
        print(f"🚀 TÌM THẤY TASK: {task['filename']} (ID: {task['_id']})")
        
        # 2. Tải ảnh từ MongoDB về máy
        print("⬇️ Đang tải ảnh từ MongoDB...")
        with open(TEMP_INPUT, "wb") as f:
            f.write(fs.get(task['_id']).read())
            
        # 3. Gọi lệnh AI để xử lý (Giống hệt lệnh bạn chạy tay)
        print("🧠 Đang chạy AI xử lý...")
        # Lưu ý: Lệnh này gọi file save_disparity.py với input là ảnh vừa tải về
        cmd = f'accelerate launch save_disparity.py model=igev_stereo checkpoint="checkpoints/igev_stereo/model_700.safetensors" left_list="{TEMP_INPUT}" right_list="{TEMP_INPUT}" disp_dir="{TEMP_OUTPUT_DIR}"'
        os.system(cmd)
        
        # 4. Kiểm tra xem có kết quả không và lưu ngược lại MongoDB
        if os.path.exists(TEMP_RESULT):
            print("⬆️ Đang upload kết quả lên MongoDB...")
            with open(TEMP_RESULT, "rb") as f:
                # Lưu file kết quả vào GridFS
                fs.put(f, filename=f"result_{task['filename']}", parent_id=task['_id'], type="disparity_map")
            
            # 5. Cập nhật trạng thái ảnh gốc thành 'done'
            collection.update_one({"_id": task['_id']}, {"$set": {"status": "done"}})
            print("✅ HOÀN THÀNH TASK! Đã lưu kết quả vào DB.")
        else:
            print("❌ LỖI: Không tìm thấy file kết quả từ AI.")
            
    else:
        print("zzz Kho dữ liệu trống (Không còn ảnh 'waiting')...")

if __name__ == "__main__":
    process_one_image()