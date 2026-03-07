"""
Data Preprocessing Tools for Strawberry Dataset
Xử lý tiền xử lý dữ liệu ảnh quả dâu tây
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt


def resize_with_padding(image, target_size=(224, 224), padding_color=(0, 0, 0)):
    """
    Resize ảnh về kích thước cố định với padding để giữ nguyên tỷ lệ khung hình
    
    Args:
        image: Ảnh đầu vào (numpy array)
        target_size: Kích thước đích (width, height), mặc định 224x224
        padding_color: Màu padding (B, G, R), mặc định màu đen (0, 0, 0)
        
    Returns:
        Ảnh đã được resize với padding
    """
    h, w = image.shape[:2]
    target_w, target_h = target_size
    
    # Tính tỷ lệ scale để giữ nguyên aspect ratio
    scale = min(target_w / w, target_h / h)
    
    # Tính kích thước mới sau khi scale
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize ảnh với tỷ lệ mới
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Tạo ảnh mới với kích thước target và fill bằng màu padding
    padded_image = np.full((target_h, target_w, 3), padding_color, dtype=np.uint8)
    
    # Tính vị trí để center ảnh đã resize
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    
    # Đặt ảnh đã resize vào giữa ảnh padding
    padded_image[top:top + new_h, left:left + new_w] = resized_image
    
    return padded_image


def process_strawberry_dataset(input_dir, output_dir, target_size=(224, 224), 
                               padding_color=(0, 0, 0), visualize_samples=True):
    """
    Xử lý toàn bộ dataset ảnh dâu tây
    
    Args:
        input_dir: Thư mục chứa ảnh gốc
        output_dir: Thư mục lưu ảnh đã xử lý
        target_size: Kích thước đích (width, height)
        padding_color: Màu padding (B, G, R)
        visualize_samples: Hiển thị mẫu ảnh trước và sau xử lý
    """
    # Tạo thư mục output nếu chưa có
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách các file ảnh
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ Thư mục không tồn tại: {input_dir}")
        return
    
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ Không tìm thấy ảnh nào trong thư mục: {input_dir}")
        return
    
    print(f"📂 Tìm thấy {len(image_files)} ảnh trong {input_dir}")
    print(f"🎯 Kích thước đích: {target_size[0]}x{target_size[1]} pixels")
    print(f"🎨 Màu padding: {padding_color}")
    print("─" * 60)
    
    # Danh sách để lưu mẫu ảnh để visualize
    samples_before = []
    samples_after = []
    
    # Xử lý từng ảnh
    processed_count = 0
    for idx, image_file in enumerate(image_files, 1):
        try:
            # Đọc ảnh
            image = cv2.imread(str(image_file))
            
            if image is None:
                print(f"⚠️  Không thể đọc ảnh: {image_file.name}")
                continue
            
            original_shape = image.shape[:2]
            
            # Resize với padding
            processed_image = resize_with_padding(image, target_size, padding_color)
            
            # Lưu ảnh đã xử lý
            output_file = output_path / image_file.name
            cv2.imwrite(str(output_file), processed_image)
            
            processed_count += 1
            print(f"✅ [{idx}/{len(image_files)}] {image_file.name}: "
                  f"{original_shape[1]}x{original_shape[0]} → "
                  f"{target_size[0]}x{target_size[1]}")
            
            # Lưu mẫu để visualize (3 ảnh đầu tiên)
            if visualize_samples and len(samples_before) < 3:
                samples_before.append((cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                                      image_file.name))
                samples_after.append((cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB),
                                     image_file.name))
                
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {image_file.name}: {str(e)}")
    
    print("─" * 60)
    print(f"✨ Hoàn thành! Đã xử lý {processed_count}/{len(image_files)} ảnh")
    print(f"📁 Ảnh đã lưu tại: {output_dir}")
    
    # Hiển thị mẫu ảnh
    if visualize_samples and samples_before:
        visualize_preprocessing(samples_before, samples_after)


def visualize_preprocessing(samples_before, samples_after):
    """
    Hiển thị ảnh trước và sau xử lý
    """
    n_samples = len(samples_before)
    fig, axes = plt.subplots(2, n_samples, figsize=(5 * n_samples, 10))
    
    if n_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(n_samples):
        # Ảnh gốc
        axes[0, i].imshow(samples_before[i][0])
        axes[0, i].set_title(f'Original: {samples_before[i][1]}\n'
                            f'Shape: {samples_before[i][0].shape[:2]}', 
                            fontsize=10)
        axes[0, i].axis('off')
        
        # Ảnh sau xử lý
        axes[1, i].imshow(samples_after[i][0])
        axes[1, i].set_title(f'Processed\n'
                            f'Shape: {samples_after[i][0].shape[:2]}', 
                            fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 Đã lưu hình ảnh so sánh: preprocessing_comparison.png")
    plt.show()


if __name__ == "__main__":
    # Ví dụ sử dụng
    
    # Đường dẫn thư mục
    INPUT_DIR = "./dataset/strawberry/Ripe"
    OUTPUT_DIR = "./dataset/strawberry/Ripe_processed_224"
    
    # Cấu hình
    TARGET_SIZE = (224, 224)  # Hoặc (299, 299) cho InceptionV3, Xception
    PADDING_COLOR = (0, 0, 0)  # Màu đen, có thể dùng (255, 255, 255) cho màu trắng
    
    # Xử lý dataset
    process_strawberry_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        target_size=TARGET_SIZE,
        padding_color=PADDING_COLOR,
        visualize_samples=True
    )