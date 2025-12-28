import os
import glob
from pyspark.sql import SparkSession
import torch
from diffusers import StableDiffusionInpaintPipeline
# Import các hàm cần thiết từ project của bạn
# Lưu ý: Bạn cần copy các file model/, util/, config/ vào cùng thư mục chạy Spark
# Hoặc đóng gói chúng thành file .zip để gửi kèm Spark job

# --- CẤU HÌNH ---
INPUT_DIR = "D:/Data/bdd100k_dataset/bdd100k/images/100k/train" # Đường dẫn dataset 10GB
OUTPUT_DIR = "D:/Data/output_spark_mapreduce"
CHECKPOINT_DIR = "checkpoints/StereoGen" # Đường dẫn model đã tải

# Hàm xử lý (MAPPER) - Chạy trên từng Worker node
def process_partition(iterator):
    """
    Hàm này nhận vào một danh sách các đường dẫn ảnh (Iterator),
    Load model AI một lần duy nhất, sau đó xử lý hết danh sách đó.
    """
    results = []
    
    # 1. Load Model (Chỉ load 1 lần trên mỗi Partition để tiết kiệm RAM/Time)
    # Lưu ý: Cần import thư viện bên trong hàm để tránh lỗi serialize của Spark
    import torch
    from diffusers import StableDiffusionInpaintPipeline
    
    # Giả lập load model (Thay bằng code load model thật của bạn trong generate_stereo.py)
    # Vì Spark chạy đa luồng, cần cẩn thận với CUDA. 
    # Nếu chạy local mode, model sẽ tranh nhau GPU. Tốt nhất set device='cpu' hoặc giới hạn worker.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        # Ở đây mình demo load pipeline SD đơn giản. 
        # Thực tế bạn cần bê logic load model từ 'generate_stereo.py' vào đây.
        print(f"Worker đang load model trên thiết bị: {device}")
        # model = ... (Load model ZeroStereo của bạn tại đây)
        
        # 2. Lặp qua từng ảnh trong partition được giao
        for image_path in iterator:
            filename = os.path.basename(image_path)
            try:
                # --- LOGIC XỬ LÝ ẢNH (Inference) ---
                # image = read_image(image_path)
                # result = model(image)
                # save_image(result, OUTPUT_DIR + filename)
                
                # Giả lập xử lý xong
                status = "SUCCESS"
                
            except Exception as e:
                status = f"FAILED: {str(e)}"
            
            # Trả về kết quả dạng Key-Value cho bước Reduce
            yield (filename, status)
            
    except Exception as e:
        yield ("System_Error", str(e))

# --- CHƯƠNG TRÌNH CHÍNH (DRIVER) ---
if __name__ == "__main__":
    # 1. Khởi tạo Spark Session
    spark = SparkSession.builder \
        .appName("ZeroStereo_MapReduce") \
        .config("spark.driver.memory", "4g") \
        .master("local[*]") \
        .getOrCreate()
        # local[*] nghĩa là dùng tất cả CPU core của máy bạn làm worker

    print("🚀 Bắt đầu Job MapReduce ZeroStereo...")

    # 2. Đọc danh sách file (Tạo RDD)
    # Tìm tất cả file jpg trong thư mục 10GB
    all_files = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    # Chỉ lấy thử 100 ảnh để test trước khi chạy full
    all_files = all_files[:100] 
    
    # Phân tán danh sách file vào RDD (Resilient Distributed Dataset)
    # numSlices=4 chia dữ liệu thành 4 phần cho 4 worker xử lý song song
    files_rdd = spark.sparkContext.parallelize(all_files, numSlices=4)

    # 3. Giai đoạn MAP: Xử lý ảnh song song
    # mapPartitions hiệu quả hơn map() vì load model 1 lần/batch
    mapped_rdd = files_rdd.mapPartitions(process_partition)

    # 4. Giai đoạn REDUCE: Tổng hợp kết quả
    # Gom các trạng thái lại và đếm
    # (Key, Value) -> Đếm số lượng Success/Failed
    summary = mapped_rdd.map(lambda x: (x[1].split(':')[0], 1)) \
                        .reduceByKey(lambda a, b: a + b) \
                        .collect()

    print("-" * 40)
    print("📊 KẾT QUẢ MAP REDUCE:")
    for status, count in summary:
        print(f"   - {status}: {count} ảnh")
    print("-" * 40)

    spark.stop()