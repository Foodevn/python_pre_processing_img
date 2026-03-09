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


def normalize_image(image):
    """
    Normalize pixel values về [0, 1] bằng cách chia cho 255.

    Args:
        image: Ảnh đầu vào (numpy array, uint8)

    Returns:
        Ảnh đã normalize (numpy array, float32, giá trị trong [0.0, 1.0])
    """
    return image.astype(np.float32) / 255.0


def process_strawberry_dataset(input_dir, output_dir, target_size=(224, 224),
                               padding_color=(0, 0, 0), color_space="BGR",
                               normalize=False, save_as_npy=False,
                               visualize_samples=True,
                               output_prefix="strawberry", add_ripe_in_name=None):
    """
    Xử lý toàn bộ dataset ảnh dâu tây
    
    Args:
        input_dir: Thư mục chứa ảnh gốc
        output_dir: Thư mục lưu ảnh đã xử lý
        target_size: Kích thước đích (width, height)
        padding_color: Màu padding (B, G, R)
        color_space: Không gian màu đầu ra 'BGR' | 'RGB' | 'HSV' | 'LAB'
        normalize: True/False.
            - True: normalize pixel values về [0.0, 1.0] (float32)
            - False: giữ nguyên giá trị uint8 [0, 255]
        save_as_npy: True/False.
            - True: lưu tensor dưới dạng .npy
            - False: lưu ảnh thông thường (nếu normalize=True sẽ chuyển lại uint8)
        visualize_samples: Hiển thị mẫu ảnh trước và sau xử lý
        output_prefix: Tiền tố tên file output
        add_ripe_in_name: True/False/None.
            - None: tự động thêm 'ripe' khi tên thư mục input là 'Ripe'
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
    
    image_files = sorted(
        [f for f in input_path.iterdir() if f.suffix.lower() in image_extensions],
        key=lambda p: p.name.lower()
    )
    
    if not image_files:
        print(f"❌ Không tìm thấy ảnh nào trong thư mục: {input_dir}")
        return
    
    print(f"📂 Tìm thấy {len(image_files)} ảnh trong {input_dir}")
    print(f"🎯 Kích thước đích: {target_size[0]}x{target_size[1]} pixels")
    print(f"🎨 Màu padding: {padding_color}")
    print(f"🌈 Color space: {color_space}")
    print(f"📐 Normalize: {'Bật (float32, [0.0, 1.0])' if normalize else 'Tắt (uint8, [0, 255])'}")
    print("─" * 60)

    color_space = color_space.upper()
    if color_space not in {"BGR", "RGB", "HSV", "LAB"}:
        raise ValueError("color_space phải là 'BGR', 'RGB', 'HSV' hoặc 'LAB'.")

    if add_ripe_in_name is None:
        # Chỉ gắn 'ripe' khi tên folder là đúng 'ripe' (tránh nhầm với 'unripe').
        add_ripe_in_name = input_path.name.lower() == "ripe"

    print(f"📝 Đặt tên tăng dần: {output_prefix}_{'ripe_' if add_ripe_in_name else ''}0001...")
    
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
            
            # Bước 1: Resize với padding
            resized_image = resize_with_padding(image, target_size, padding_color)
            
            # Bước 4: Chuyển đổi không gian màu (tùy chọn)
            processed_image = convert_color_space(resized_image, color_space=color_space)

            # Bước 5: Normalize pixel values (tùy chọn)
            if normalize:
                processed_image = normalize_image(processed_image)

            # Lưu ảnh đã xử lý
            name_stem = f"{output_prefix}_{'ripe_' if add_ripe_in_name else ''}{idx:04d}"
            if save_as_npy:
                output_file = output_path / f"{name_stem}.npy"
                np.save(str(output_file), processed_image)
            else:
                output_file = output_path / f"{name_stem}{image_file.suffix.lower()}"
                # Nếu đã normalize thì chuyển lại uint8 trước khi lưu ảnh
                img_to_save = (processed_image * 255).astype(np.uint8) if normalize else processed_image
                save_image = to_save_bgr(img_to_save, color_space)
                cv2.imwrite(str(output_file), save_image)
            
            processed_count += 1
            print(f"✅ [{idx}/{len(image_files)}] {image_file.name}: "
                  f"{original_shape[1]}x{original_shape[0]} → "
                  f"{target_size[0]}x{target_size[1]} | {color_space}")
            
            # Lưu mẫu để visualize (3 ảnh đầu tiên)
            if visualize_samples and len(samples_before) < 3:
                samples_before.append((cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                                      image_file.name))
                # Chuyển về uint8 nếu đã normalize để hiển thị đúng màu
                display_image = (processed_image * 255).astype(np.uint8) if normalize else processed_image
                samples_after.append((to_display_rgb(display_image, color_space),
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

if __name__ == "__main__":
    # Ví dụ sử dụng
    process_strawberry_dataset(
        input_dir="./dataset/strawberry/Ripe",                 # Thư mục ảnh gốc (label: ripe)
        output_dir="./dataset/strawberry/Ripe_preprocessed",  # Thư mục lưu ảnh sau xử lý
        target_size=(224, 224),                                 # Kích thước chuẩn cho CNN
        padding_color=(0, 0, 0),                                # Màu viền khi giữ tỉ lệ ảnh
        color_space="HSV",                                     # Không gian màu đầu ra: BGR/RGB/HSV/LAB
        normalize=False,                                        # True: float32 [0,1], False: uint8 [0,255]
        save_as_npy=False,                                      # False: lưu ảnh, True: lưu tensor .npy
        visualize_samples=True,                                 # Hiển thị ảnh trước/sau (3 mẫu đầu)
        output_prefix="strawberry",                            # Tiền tố tên file output
        add_ripe_in_name=None                                   # None: tự nhận diện folder Ripe để thêm 'ripe'
    )
    process_strawberry_dataset(
        input_dir="./dataset/strawberry/Unripe",                 # Thư mục ảnh gốc (label: unripe)
        output_dir="./dataset/strawberry/Unripe_preprocessed",  # Thư mục lưu ảnh sau xử lý
        target_size=(224, 224),                                   # Kích thước chuẩn cho CNN
        padding_color=(0, 0, 0),                                  # Màu viền khi giữ tỉ lệ ảnh
        color_space="HSV",                                       # Không gian màu đầu ra: BGR/RGB/HSV/LAB
        normalize=False,                                          # True: float32 [0,1], False: uint8 [0,255]
        save_as_npy=False,                                        # False: lưu ảnh, True: lưu tensor .npy
        visualize_samples=True,                                   # Hiển thị ảnh trước/sau (3 mẫu đầu)
        output_prefix="strawberry_unripe",                       # Tiền tố tên file output
        add_ripe_in_name=None                                     # None: không thêm 'ripe' cho folder Unripe
    )   
 