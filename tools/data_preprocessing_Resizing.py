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


def convert_color_space(image, color_space="BGR"):
    """
    Bước 4 (tùy chọn): Chuyển đổi không gian màu.

    Args:
        image: Ảnh đầu vào theo BGR (cv2.imread)
        color_space: 'BGR' | 'RGB' | 'HSV' | 'LAB'

    Returns:
        Ảnh đã chuyển đổi theo không gian màu yêu cầu
    """
    color_space = color_space.upper()

    if color_space == "BGR":
        return image
    if color_space == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if color_space == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if color_space == "LAB":
        return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    raise ValueError("color_space phải là 'BGR', 'RGB', 'HSV' hoặc 'LAB'.")


def to_display_rgb(image, color_space):
    """
    Chuyển ảnh đã xử lý về RGB để hiển thị bằng matplotlib.
    """
    color_space = color_space.upper()

    if color_space == "BGR":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if color_space == "RGB":
        return image
    if color_space == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    if color_space == "LAB":
        return cv2.cvtColor(image, cv2.COLOR_LAB2RGB)

    raise ValueError("color_space phải là 'BGR', 'RGB', 'HSV' hoặc 'LAB'.")


def to_save_bgr(image, color_space):
    """
    Chuyển ảnh về BGR để lưu bằng cv2.imwrite.
    """
    color_space = color_space.upper()

    if color_space == "BGR":
        return image
    if color_space == "RGB":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if color_space == "HSV":
        return cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    if color_space == "LAB":
        return cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

    raise ValueError("color_space phải là 'BGR', 'RGB', 'HSV' hoặc 'LAB'.")


def process_strawberry_dataset(input_dir, output_dir, target_size=(224, 224), 
                               padding_color=(0, 0, 0), color_space="BGR",
                               save_as_npy=False, visualize_samples=True):
    """
    Xử lý toàn bộ dataset ảnh dâu tây
    
    Args:
        input_dir: Thư mục chứa ảnh gốc
        output_dir: Thư mục lưu ảnh đã xử lý
        target_size: Kích thước đích (width, height)
        padding_color: Màu padding (B, G, R)
        color_space: Không gian màu đầu ra 'BGR' | 'RGB' | 'HSV' | 'LAB'
        save_as_npy: True/False.
            - True: lưu tensor dưới dạng .npy
            - False: lưu ảnh thông thường
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
    print(f"🌈 Color space: {color_space}")
    print("─" * 60)

    color_space = color_space.upper()
    if color_space not in {"BGR", "RGB", "HSV", "LAB"}:
        raise ValueError("color_space phải là 'BGR', 'RGB', 'HSV' hoặc 'LAB'.")
    
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
            resized_image = resize_with_padding(image, target_size, padding_color)
            
            # Bước 4: Chuyển đổi không gian màu (tùy chọn)
            processed_image = convert_color_space(resized_image, color_space=color_space)
            
            # Lưu ảnh đã xử lý
            if save_as_npy:
                output_file = output_path / f"{image_file.stem}.npy"
                np.save(str(output_file), processed_image)
            else:
                output_file = output_path / image_file.name
                save_image = to_save_bgr(processed_image, color_space)
                cv2.imwrite(str(output_file), save_image)
            
            processed_count += 1
            print(f"✅ [{idx}/{len(image_files)}] {image_file.name}: "
                  f"{original_shape[1]}x{original_shape[0]} → "
                  f"{target_size[0]}x{target_size[1]} | {color_space}")
            
            # Lưu mẫu để visualize (3 ảnh đầu tiên)
            if visualize_samples and len(samples_before) < 3:
                samples_before.append((cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                                      image_file.name))
                samples_after.append((to_display_rgb(processed_image, color_space),
                                     image_file.name))
                
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {image_file.name}: {str(e)}")
    
    print("─" * 60)
    print(f"✨ Hoàn thành! Đã xử lý {processed_count}/{len(image_files)} ảnh")
    print(f"📁 Ảnh đã lưu tại: {output_dir}")
    if save_as_npy:
        print("💾 Định dạng lưu: .npy (phù hợp làm input cho model)")
    
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
    INPUT_DIR = "./dataset/strawberry/Unripe"
    OUTPUT_DIR = "./dataset/strawberry/Unripe_processed_224"
    
    # Cấu hình
    TARGET_SIZE = (224, 224)  # Hoặc (299, 299) cho InceptionV3, Xception
    PADDING_COLOR = (0, 0, 0)  # Màu đen, có thể dùng (255, 255, 255) cho màu trắng
    COLOR_SPACE = "HSV"  # 'BGR' | 'RGB' | 'HSV' | 'LAB'
    SAVE_AS_NPY = False   # Bạn chưa cần xuất .npy
    
    # Xử lý dataset
    process_strawberry_dataset(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        target_size=TARGET_SIZE,
        padding_color=PADDING_COLOR,
        color_space=COLOR_SPACE,
        save_as_npy=SAVE_AS_NPY,
        visualize_samples=True
    )