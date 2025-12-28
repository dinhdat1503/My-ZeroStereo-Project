import pymongo
import gridfs
import os

# 1. Kết nối đến MongoDB (Mặc định chạy trên máy local)
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["BigDataStereo"] # Tên cơ sở dữ liệu (tự đặt)
fs = gridfs.GridFS(db)       # Dùng GridFS để lưu file ảnh lớn

# 2. Đường dẫn ảnh cần nạp (Dùng ảnh demo lúc nãy)
image_path = "assets/demo.jpg"

if os.path.exists(image_path):
    # 3. Đọc file ảnh và đẩy lên MongoDB
    with open(image_path, "rb") as f:
        # Lưu vào GridFS, đặt tên file là 'input_image.jpg'
        file_id = fs.put(f, filename="input_image.jpg", status="waiting")
        
    print("------------------------------------------------")
    print("✅ ĐÃ NẠP ẢNH THÀNH CÔNG!")
    print(f"📁 Database: BigDataStereo")
    print(f"🔑 ID của ảnh: {file_id}")
    print("------------------------------------------------")
else:
    print(f"❌ Không tìm thấy file ảnh tại: {image_path}")