import os
import glob
import sys
from pyspark.sql import SparkSession

# --- CẤU HÌNH CHO KAGGLE ---
# Lưu ý: Bạn cần thay đúng tên dataset của bạn ở dòng dưới
INPUT_DIR = "/kaggle/input/bdd100k-dataset/bdd100k/images/100k/train" 
OUTPUT_DIR = "/kaggle/working/spark_output"

# Hàm Map (Xử lý từng phần tử)
def process_partition(iterator):
    import torch # Import bên trong hàm để tránh lỗi Spark
    results = []
    
    # Giả lập check GPU trên Worker
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for image_path in iterator:
        filename = os.path.basename(image_path)
        try:
            # --- CHỖ NÀY GỌI MODEL CỦA BẠN ---
            # Ví dụ: model.predict(image_path)
            # Ở đây mình chỉ giả lập tạo file kết quả rỗng
            
            # Ghi file kết quả (Demo)
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(f"{OUTPUT_DIR}/{filename}.txt", "w") as f:
                f.write(f"Processed on {device}")
                
            status = "SUCCESS"
        except Exception as e:
            status = "FAILED"
        
        yield (filename, status)

if __name__ == "__main__":
    # Khởi tạo Spark (Chế độ local[2] cho Kaggle T4 x2)
    spark = SparkSession.builder \
        .appName("ZeroStereo_MapReduce") \
        .config("spark.driver.memory", "14g") \
        .master("local[2]") \
        .getOrCreate()

    print(f"🚀 [Spark] Đang quét ảnh từ: {INPUT_DIR}")
    
    # Kiểm tra folder đầu vào
    if not os.path.exists(INPUT_DIR):
        print(f"❌ LỖI: Không tìm thấy đường dẫn {INPUT_DIR}")
        print("👉 Hãy kiểm tra lại tên Dataset trong phần 'Add Input' trên Kaggle!")
        sys.exit(1)

    all_files = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    print(f"📊 Tổng số ảnh tìm thấy: {len(all_files)}")

    # CHẠY TEST 100 ẢNH
    if len(all_files) > 0:
        files_rdd = spark.sparkContext.parallelize(all_files[:100], numSlices=2)
        
        print("⏳ Đang chạy MapReduce...")
        # Map: Xử lý
        mapped_rdd = files_rdd.mapPartitions(process_partition)
        
        # Reduce: Tổng hợp
        summary = mapped_rdd.map(lambda x: (x[1], 1)) \
                            .reduceByKey(lambda a, b: a + b) \
                            .collect()

        print("-" * 40)
        print("✅ KẾT QUẢ THỐNG KÊ (REDUCE):")
        for status, count in summary:
            print(f"   - {status}: {count}")
        print("-" * 40)
    
    spark.stop()