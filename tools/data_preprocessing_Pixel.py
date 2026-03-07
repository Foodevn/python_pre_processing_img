"""
Data Preprocessing Tools - Pixel Normalization
Chuẩn hóa giá trị pixel cho ảnh dâu tây
"""

import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def rescaling(image):
    """
    Rescaling: Chia giá trị pixel cho 255 để đưa về khoảng [0, 1]
    
    Args:
        image: Ảnh đầu vào (numpy array) với giá trị pixel từ 0-255
        
    Returns:
        Ảnh đã được rescale về khoảng [0, 1]
    """
    return image.astype(np.float32) / 255.0


def standardization(image, mean=None, std=None):
    """
    Standardization: Chuẩn hóa theo công thức (X - μ) / σ
    
    Args:
        image: Ảnh đầu vào (numpy array)
        mean: Giá trị trung bình (nếu None, tính từ ảnh)
        std: Độ lệch chuẩn (nếu None, tính từ ảnh)
        
    Returns:
        Ảnh đã được standardize, mean, std
    """
    # Chuyển về float32
    img_float = image.astype(np.float32)
    
    # Tính mean và std nếu chưa có
    if mean is None:
        mean = np.mean(img_float, axis=(0, 1), keepdims=True)
    
    if std is None:
        std = np.std(img_float, axis=(0, 1), keepdims=True)
        # Tránh chia cho 0
        std = np.where(std == 0, 1, std)
    
    # Standardization: (X - μ) / σ
    standardized = (img_float - mean) / std
    
    return standardized, mean, std


def imagenet_standardization(image):
    """
    Standardization theo ImageNet (phổ biến cho transfer learning)
    
    ImageNet mean (RGB): [0.485, 0.456, 0.406]
    ImageNet std (RGB): [0.229, 0.224, 0.225]
    
    Args:
        image: Ảnh đầu vào BGR (từ cv2.imread)
        
    Returns:
        Ảnh đã được standardize theo chuẩn ImageNet
    """
    # Chuyển BGR sang RGB
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Rescale về [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # ImageNet mean và std (RGB)
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    
    # Standardization
    standardized = (img_float - imagenet_mean) / imagenet_std
    
    return standardized


def process_image_normalization(image_path, method='rescaling', 
                                imagenet_stats=False):
    """
    Xử lý một ảnh với phương pháp chuẩn hóa được chọn
    
    Args:
        image_path: Đường dẫn đến ảnh
        method: 'rescaling' hoặc 'standardization'
        imagenet_stats: Nếu True, dùng ImageNet mean/std
        
    Returns:
        Ảnh gốc, ảnh đã chuẩn hóa, thông tin thống kê
    """
    # Đọc ảnh
    image = cv2.imread(str(image_path))
    
    if image is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    stats = {}
    
    if method == 'rescaling':
        normalized = rescaling(image)
        stats['method'] = 'Rescaling: X / 255'
        stats['range'] = f"[{normalized.min():.4f}, {normalized.max():.4f}]"
        stats['mean'] = normalized.mean()
        stats['std'] = normalized.std()
        
    elif method == 'standardization':
        if imagenet_stats:
            normalized = imagenet_standardization(image)
            stats['method'] = 'ImageNet Standardization'
            stats['mean'] = 'ImageNet: [0.485, 0.456, 0.406]'
            stats['std'] = 'ImageNet: [0.229, 0.224, 0.225]'
        else:
            normalized, mean, std = standardization(image)
            stats['method'] = 'Standardization: (X - μ) / σ'
            stats['mean'] = mean.flatten()
            stats['std'] = std.flatten()
        
        stats['range'] = f"[{normalized.min():.4f}, {normalized.max():.4f}]"
    
    else:
        raise ValueError(f"Phương pháp không hợp lệ: {method}")
    
    return image, normalized, stats


def batch_normalize_dataset(input_dir, output_dir, method='rescaling',
                            imagenet_stats=False, save_as_npy=True):
    """
    Chuẩn hóa toàn bộ dataset và lưu kết quả
    
    Args:
        input_dir: Thư mục chứa ảnh gốc
        output_dir: Thư mục lưu ảnh đã chuẩn hóa
        method: 'rescaling' hoặc 'standardization'
        imagenet_stats: Dùng ImageNet statistics (cho transfer learning)
        save_as_npy: Lưu dưới dạng .npy (True) hoặc ảnh PNG (False)
    """
    # Tạo thư mục output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Lấy danh sách ảnh
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"❌ Thư mục không tồn tại: {input_dir}")
        return
    
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ Không tìm thấy ảnh nào trong: {input_dir}")
        return
    
    print(f"📂 Tìm thấy {len(image_files)} ảnh")
    print(f"🎯 Phương pháp: {method.upper()}")
    if method == 'standardization' and imagenet_stats:
        print(f"📊 Sử dụng ImageNet statistics")
    print("─" * 60)
    
    processed_count = 0
    all_normalized = []
    
    for idx, image_file in enumerate(image_files, 1):
        try:
            original, normalized, stats = process_image_normalization(
                image_file, method, imagenet_stats
            )
            
            if save_as_npy:
                # Lưu dưới dạng numpy array
                output_file = output_path / f"{image_file.stem}.npy"
                np.save(output_file, normalized)
            else:
                # Denormalize và lưu dưới dạng ảnh PNG
                if method == 'rescaling':
                    # Nhân với 255 và convert về uint8
                    denorm = (normalized * 255).astype(np.uint8)
                else:
                    # Với standardization, cần scale lại về [0, 255]
                    denorm = cv2.normalize(normalized, None, 0, 255, 
                                          cv2.NORM_MINMAX, cv2.CV_8U)
                
                output_file = output_path / f"{image_file.stem}.png"
                cv2.imwrite(str(output_file), denorm)
            
            all_normalized.append(normalized)
            processed_count += 1
            
            print(f"✅ [{idx}/{len(image_files)}] {image_file.name}")
            
        except Exception as e:
            print(f"❌ Lỗi khi xử lý {image_file.name}: {str(e)}")
    
    print("─" * 60)
    print(f"✨ Hoàn thành! Đã xử lý {processed_count}/{len(image_files)} ảnh")
    print(f"📁 Kết quả lưu tại: {output_dir}")
    
    # Tính statistics cho toàn bộ dataset
    if all_normalized:
        dataset_stats = calculate_dataset_statistics(all_normalized)
        print("\n📊 Thống kê toàn bộ dataset:")
        print(f"   Mean: {dataset_stats['mean']}")
        print(f"   Std: {dataset_stats['std']}")
        print(f"   Min: {dataset_stats['min']:.4f}")
        print(f"   Max: {dataset_stats['max']:.4f}")
        
        # Lưu statistics
        stats_file = output_path / "dataset_statistics.npz"
        np.savez(stats_file, 
                mean=dataset_stats['mean'],
                std=dataset_stats['std'],
                min=dataset_stats['min'],
                max=dataset_stats['max'])
        print(f"💾 Statistics đã lưu: {stats_file.name}")


def calculate_dataset_statistics(images):
    """
    Tính statistics cho toàn bộ dataset
    """
    # Stack tất cả ảnh
    stacked = np.stack(images, axis=0)
    
    return {
        'mean': np.mean(stacked, axis=(0, 1, 2)),
        'std': np.std(stacked, axis=(0, 1, 2)),
        'min': np.min(stacked),
        'max': np.max(stacked)
    }


def visualize_normalization(image_path, save_plot=True):
    """
    Hiển thị so sánh các phương pháp chuẩn hóa
    """
    # Đọc ảnh gốc
    original = cv2.imread(str(image_path))
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # Áp dụng các phương pháp
    rescaled = rescaling(original_rgb)
    standardized, mean, std = standardization(original_rgb)
    imagenet_std = imagenet_standardization(original)
    
    # Tạo figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Hàng 1: Hiển thị ảnh
    # Original
    axes[0, 0].imshow(original_rgb)
    axes[0, 0].set_title('Original\nRange: [0, 255]', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Rescaled - hiển thị được vì đã ở [0,1]
    axes[0, 1].imshow(rescaled)
    axes[0, 1].set_title(f'Rescaling: X / 255\nRange: [{rescaled.min():.2f}, {rescaled.max():.2f}]', 
                          fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Standardized - cần normalize để hiển thị
    std_display = cv2.normalize(standardized, None, 0, 1, cv2.NORM_MINMAX)
    axes[0, 2].imshow(std_display)
    axes[0, 2].set_title(f'Standardization: (X-μ)/σ\nRange: [{standardized.min():.2f}, {standardized.max():.2f}]', 
                          fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Hàng 2: Histogram
    colors = ['red', 'green', 'blue']
    channels = ['Red', 'Green', 'Blue']
    
    # Original histogram
    for i, (color, ch_name) in enumerate(zip(colors, channels)):
        axes[1, 0].hist(original_rgb[:, :, i].ravel(), bins=50, 
                       color=color, alpha=0.5, label=ch_name)
    axes[1, 0].set_title('Original Pixel Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Pixel Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Rescaled histogram
    for i, (color, ch_name) in enumerate(zip(colors, channels)):
        axes[1, 1].hist(rescaled[:, :, i].ravel(), bins=50, 
                       color=color, alpha=0.5, label=ch_name)
    axes[1, 1].set_title('Rescaled Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Normalized Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Standardized histogram
    for i, (color, ch_name) in enumerate(zip(colors, channels)):
        axes[1, 2].hist(standardized[:, :, i].ravel(), bins=50, 
                       color=color, alpha=0.5, label=ch_name)
    axes[1, 2].set_title('Standardized Distribution', fontweight='bold')
    axes[1, 2].set_xlabel('Standardized Value')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle('Pixel Normalization Comparison', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('pixel_normalization_comparison.png', dpi=150, bbox_inches='tight')
        print("📊 Đã lưu biểu đồ so sánh: pixel_normalization_comparison.png")
    
    plt.show()
    
    # In thống kê chi tiết
    print("\n📊 Thống kê chi tiết:")
    print("─" * 60)
    print("Original:")
    print(f"  Mean: {original_rgb.mean():.2f}, Std: {original_rgb.std():.2f}")
    print(f"  Range: [{original_rgb.min()}, {original_rgb.max()}]")
    print("\nRescaling (X / 255):")
    print(f"  Mean: {rescaled.mean():.4f}, Std: {rescaled.std():.4f}")
    print(f"  Range: [{rescaled.min():.4f}, {rescaled.max():.4f}]")
    print("\nStandardization ((X - μ) / σ):")
    print(f"  Mean: {standardized.mean():.4f}, Std: {standardized.std():.4f}")
    print(f"  Range: [{standardized.min():.4f}, {standardized.max():.4f}]")
    print("─" * 60)


if __name__ == "__main__":
    # Ví dụ 1: Visualize một ảnh với các phương pháp chuẩn hóa
    print("=" * 60)
    print("VÍ DỤ 1: VISUALIZE CÁC PHƯƠNG PHÁP CHUẨN HÓA")
    print("=" * 60)
    
    # Thay đổi đường dẫn này thành ảnh bạn muốn test
    sample_image = "./dataset/strawberry/Ripe/image_001.jpg"
    
    if Path(sample_image).exists():
        visualize_normalization(sample_image, save_plot=True)
    else:
        print(f"⚠️ File không tồn tại: {sample_image}")
        print("   Vui lòng cập nhật đường dẫn trong code")
    
    print("\n")
    
    # Ví dụ 2: Xử lý toàn bộ dataset với Rescaling
    print("=" * 60)
    print("VÍ DỤ 2: XỬ LÝ DATASET VỚI RESCALING")
    print("=" * 60)
    
    batch_normalize_dataset(
        input_dir="./dataset/strawberry/Ripe",
        output_dir="./dataset/strawberry/Ripe_rescaled",
        method='rescaling',
        save_as_npy=True  # Lưu dưới dạng .npy để dùng cho training
    )
    
    print("\n")
    
    # Ví dụ 3: Xử lý với ImageNet Standardization (cho transfer learning)
    print("=" * 60)
    print("VÍ DỤ 3: XỬ LÝ VỚI IMAGENET STANDARDIZATION")
    print("=" * 60)
    
    batch_normalize_dataset(
        input_dir="./dataset/strawberry/Ripe",
        output_dir="./dataset/strawberry/Ripe_imagenet_std",
        method='standardization',
        imagenet_stats=True,  # Quan trọng cho transfer learning
        save_as_npy=True
    )
