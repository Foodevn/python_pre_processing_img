"""
Data Preprocessing Tools for Strawberry Dataset
Xử lý tiền xử lý dữ liệu ảnh quả dâu tây
"""

import cv2
import numpy as np
import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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


def augment_image(image, augmentations=None):
    """
    Tạo các biến thể augmented từ ảnh gốc (uint8).

    Args:
        image: Ảnh đầu vào (numpy array, uint8)
        augmentations: Danh sách các phép augment muốn áp dụng.
            Hỗ trợ: 'flip_h', 'flip_v', 'rotate_15', 'rotate_neg15',
                    'brightness_up', 'brightness_down'
            Mặc định (None): áp dụng tất cả 6 phép trên.

    Returns:
        List các tuple (augmented_image, suffix) — suffix dùng để đặt tên file.
    """
    if augmentations is None:
        augmentations = ["flip_h", "flip_v", "rotate_15", "rotate_neg15",
                         "brightness_up", "brightness_down"]

    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2
    results = []

    for aug in augmentations:
        if aug == "flip_h":
            results.append((cv2.flip(image, 1), "flip_h"))
        elif aug == "flip_v":
            results.append((cv2.flip(image, 0), "flip_v"))
        elif aug == "rotate_15":
            M = cv2.getRotationMatrix2D((cx, cy), 15, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            results.append((rotated, "rot15"))
        elif aug == "rotate_neg15":
            M = cv2.getRotationMatrix2D((cx, cy), -15, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            results.append((rotated, "rot_15"))
        elif aug == "brightness_up":
            results.append((cv2.convertScaleAbs(image, alpha=1.2, beta=30), "bright_up"))
        elif aug == "brightness_down":
            results.append((cv2.convertScaleAbs(image, alpha=0.8, beta=-30), "bright_dn"))
        else:
            print(f"⚠️  Augmentation không hợp lệ, bỏ qua: '{aug}'")

    return results


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
                               normalize=False, augment=False, augmentations=None,
                               save_as_npy=False, visualize_samples=True,
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
        augment: True/False.
            - True: tạo thêm các ảnh augmented cho mỗi ảnh gốc
            - False: không augment (mặc định)
        augmentations: Danh sách augmentation muốn dùng (None = dùng tất cả mặc định).
            Ví dụ: ['flip_h', 'rotate_15', 'brightness_up']
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
    _aug_list = augmentations if augmentations else ["flip_h", "flip_v", "rotate_15", "rotate_neg15", "brightness_up", "brightness_down"]
    print(f"🔀 Augment: {'Bật → ' + ', '.join(_aug_list) if augment else 'Tắt'}")
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

            # Bước 5: Augmentation (tùy chọn) — áp dụng trước normalize
            aug_variants = []  # list of (image_uint8, suffix)
            if augment:
                aug_variants = augment_image(processed_image, augmentations)

            # Bước 6: Normalize pixel values (tùy chọn)
            if normalize:
                processed_image = normalize_image(processed_image)

            # Hàm trợ giúp lưu một ảnh (uint8 hoặc float32)
            def _save(img, stem):
                if save_as_npy:
                    np.save(str(output_path / f"{stem}.npy"), img)
                else:
                    img_uint8 = (img * 255).astype(np.uint8) if normalize else img
                    cv2.imwrite(str(output_path / f"{stem}{image_file.suffix.lower()}"),
                                to_save_bgr(img_uint8, color_space))

            # Lưu ảnh gốc đã xử lý
            name_stem = f"{output_prefix}_{'ripe_' if add_ripe_in_name else ''}{idx:04d}"
            _save(processed_image, name_stem)

            # Lưu các ảnh augmented
            for aug_img, aug_suffix in aug_variants:
                aug_normalized = normalize_image(aug_img) if normalize else aug_img
                _save(aug_normalized, f"{name_stem}_{aug_suffix}")
            
            processed_count += 1
            aug_info = f" + {len(aug_variants)} augmented" if aug_variants else ""
            print(f"✅ [{idx}/{len(image_files)}] {image_file.name}: "
                  f"{original_shape[1]}x{original_shape[0]} → "
                  f"{target_size[0]}x{target_size[1]} | {color_space}{aug_info}")
            
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
    aug_count = processed_count * len(_aug_list) if augment else 0
    print(f"✨ Hoàn thành! Đã xử lý {processed_count}/{len(image_files)} ảnh"
          + (f" → tổng {processed_count + aug_count} file (gốc + augmented)" if augment else ""))
    print(f"📁 Ảnh đã lưu tại: {output_dir}")
    if save_as_npy:
        print("💾 Định dạng lưu: .npy (phù hợp làm input cho model)")
    
    # Hiển thị mẫu ảnh
    if visualize_samples and samples_before:
        visualize_preprocessing(samples_before, samples_after)


def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                  random_state=42, copy_files=True):
    """
    Chia tập dữ liệu thành train / val / test từ nhiều thư mục class con.

    Cấu trúc source_dir:
        source_dir/
            ClassName1/   (vd: Ripe_preprocessed)
            ClassName2/   (vd: Unripe_preprocessed)

    Kết quả output_dir:
        output_dir/
            train/ClassName1/, train/ClassName2/
            val/ClassName1/,   val/ClassName2/
            test/ClassName1/,  test/ClassName2/

    Args:
        source_dir:   Thư mục gốc chứa các class con
        output_dir:   Thư mục đích để lưu train/val/test
        train_ratio:  Tỷ lệ tập huấn luyện (mặc định 0.7)
        val_ratio:    Tỷ lệ tập xác thực   (mặc định 0.2)
        test_ratio:   Tỷ lệ tập kiểm tra   (mặc định 0.1)
        random_state: Seed ngẫu nhiên để tái lập kết quả
        copy_files:   True = copy file, False = chỉ tạo symlink (tiết kiệm disk)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio phải bằng 1.0")

    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if not source_path.exists():
        print(f"❌ Thư mục không tồn tại: {source_dir}")
        return

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.npy'}
    class_dirs = sorted([d for d in source_path.iterdir() if d.is_dir()])

    if not class_dirs:
        print(f"❌ Không tìm thấy thư mục class nào trong: {source_dir}")
        return

    print(f"📂 Source: {source_dir}")
    print(f"📁 Output: {output_dir}")
    print(f"📊 Tỷ lệ chia: train={train_ratio:.0%} | val={val_ratio:.0%} | test={test_ratio:.0%}")
    print(f"🎲 Random state: {random_state}")
    print("─" * 60)

    total_stats = {"train": 0, "val": 0, "test": 0}

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = sorted(
            [f for f in class_dir.iterdir() if f.suffix.lower() in image_extensions],
            key=lambda p: p.name.lower()
        )

        if not files:
            print(f"⚠️  Bỏ qua '{class_name}': không có file ảnh")
            continue

        # Chia train / temp, rồi chia temp → val / test
        val_test_ratio = val_ratio + test_ratio
        test_in_temp = test_ratio / val_test_ratio

        train_files, temp_files = train_test_split(
            files, test_size=val_test_ratio, random_state=random_state
        )
        val_files, test_files = train_test_split(
            temp_files, test_size=test_in_temp, random_state=random_state
        )

        splits = {"train": train_files, "val": val_files, "test": test_files}

        for split_name, split_files in splits.items():
            dest_dir = output_path / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for f in split_files:
                dest = dest_dir / f.name
                if copy_files:
                    shutil.copy2(str(f), str(dest))
                else:
                    if not dest.exists():
                        dest.symlink_to(f.resolve())

            total_stats[split_name] += len(split_files)

        print(f"  📂 {class_name:30s} → "
              f"train: {len(train_files):4d} | "
              f"val: {len(val_files):4d} | "
              f"test: {len(test_files):4d} "
              f"(tổng: {len(files)})")

    print("─" * 60)
    print(f"✨ Hoàn thành! Tổng: "
          f"train={total_stats['train']} | "
          f"val={total_stats['val']} | "
          f"test={total_stats['test']}")
    print(f"📁 Đã lưu tại: {output_dir}")
    # Bước 1: Tiền xử lý ảnh
    process_strawberry_dataset(
        input_dir="./dataset/strawberry/Ripe",
        output_dir="./dataset/strawberry/Ripe_preprocessed",
        target_size=(224, 224),
        padding_color=(0, 0, 0),
        color_space="HSV",
        normalize=True,
        augment=True,
        augmentations=None,
        save_as_npy=False,
        visualize_samples=True,
        output_prefix="strawberry",
        add_ripe_in_name=None
    )
    process_strawberry_dataset(
        input_dir="./dataset/strawberry/Unripe",
        output_dir="./dataset/strawberry/Unripe_preprocessed",
        target_size=(224, 224),
        padding_color=(0, 0, 0),
        color_space="HSV",
        normalize=False,
        augment=False,
        augmentations=None,
        save_as_npy=False,
        visualize_samples=True,
        output_prefix="strawberry_unripe",
        add_ripe_in_name=None
    )

    # Bước 2: Chia train/val/test (70% / 20% / 10%)
    split_dataset(
        source_dir="./dataset/strawberry",         # Thư mục chứa Ripe_preprocessed & Unripe_preprocessed
        output_dir="./dataset/strawberry/split",   # Kết quả: split/train, split/val, split/test
        train_ratio=0.7,
        val_ratio=0.2,
        test_ratio=0.1,
        random_state=42,
        copy_files=True                            # False: dùng symlink để tiết kiệm dung lượng
    )