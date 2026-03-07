"""
Training Model for Strawberry Ripeness Classification (Ripe vs Unripe)
Huấn luyện mô hình phân loại trái dâu chín và chưa chín

PHƯƠNG PHÁP & THUẬT TOÁN:
═══════════════════════════════════════════════════════════════════

1. TRANSFER LEARNING (Học Chuyển Giao)
   - Sử dụng model pre-trained trên ImageNet (đã học trên 1.2M ảnh)
   - Chỉ huấn luyện lại layer cuối (Fine-tuning)
   - Lợi: Tốc độ nhanh, cần ít dữ liệu, độ chính xác cao

2. DATA AUGMENTATION (Tăng Cường Dữ Liệu)
   - Xoay ảnh, zoom, flip, thay đổi độ sáng
   - Tránh overfitting, tăng độ generalizable

3. CNN BACKBONE: MobileNetV2
   - Nhẹ (3.5M params), nhanh, phù hợp production
   - Có thể thay bằng: ResNet50, EfficientNet, InceptionV3

4. LOSS FUNCTION: Binary Crossentropy
   - Phù hợp cho bài toán 2 lớp (ripe / unripe)
   - Công thức: -[y*log(ŷ) + (1-y)*log(1-ŷ)]

5. OPTIMIZER: Adam
   - Adaptive learning rate, hội tụ nhanh
   - Thích hợp cho các bộ dữ liệu nhỏ-vừa

6. REGULARIZATION:
   - Batch Normalization: Chuẩn hóa input mỗi layer → hội tụ nhanh
   - Dropout: Tắt 50% neuron ngẫu nhiên → tránh overfitting
   - Early Stopping: Dừng khi validation loss không cải thiện

7. METRICS:
   - Accuracy: (TP+TN)/(TP+TN+FP+FN)
   - Precision: TP/(TP+FP) - Chính xác trong dự đoán dương
   - Recall: TP/(TP+FN) - Tìm được bao nhiêu trái chín

═══════════════════════════════════════════════════════════════════
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════
# CẤU HÌNH
# ═══════════════════════════════════════════════════════════════════

CONFIG = {
    # Dữ liệu
    'ripe_dir': '../dataset/strawberry/Ripe_processed_224',
    'unripe_dir': '../dataset/strawberry/Unripe_processed_224',
    'test_size': 0.2,  # 20% dùng test, 80% dùng train
    'val_split': 0.2,  # 20% trong train dùng validation
    
    # Model
    'img_size': (224, 224),  # MobileNetV2 yêu cầu 224x224
    'batch_size': 16,
    'epochs': 20,
    'learning_rate': 0.001,
    
    # Output
    'model_save_path': './models/strawberry_ripeness_model.h5',
    'history_plot_path': './plots/training_history.png',
}


# ═══════════════════════════════════════════════════════════════════
# 1. LOAD & CHUẨN BỊ DỮ LIỆU
# ═══════════════════════════════════════════════════════════════════

def load_and_split_data(ripe_dir, unripe_dir, test_size=0.2, random_state=42):
    """
    Load ảnh từ 2 thư mục, chia train/test
    """
    from sklearn.model_selection import train_test_split
    
    ripe_dir = Path(ripe_dir)
    unripe_dir = Path(unripe_dir)
    
    # Lấy danh sách file
    ripe_files = sorted([str(f) for f in ripe_dir.glob('**/*') 
                        if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}])
    unripe_files = sorted([str(f) for f in unripe_dir.glob('**/*')
                          if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}])
    
    print(f"📂 Tìm thấy:")
    print(f"   - Ripe (chín): {len(ripe_files)} ảnh")
    print(f"   - Unripe (chưa chín): {len(unripe_files)} ảnh")
    print(f"   - Tổng cộng: {len(ripe_files) + len(unripe_files)} ảnh")
    print("─" * 60)
    
    # Gán label: 1 = ripe, 0 = unripe
    ripe_labels = [1] * len(ripe_files)
    unripe_labels = [0] * len(unripe_files)
    
    all_files = ripe_files + unripe_files
    all_labels = ripe_labels + unripe_labels
    
    # Chia train/test
    train_files, test_files, train_labels, test_labels = train_test_split(
        all_files, all_labels, test_size=test_size, random_state=random_state, stratify=all_labels
    )
    
    print(f"✂️  Chia dữ liệu:")
    print(f"   - Train: {len(train_files)} ảnh ({100*(1-test_size):.0f}%)")
    print(f"   - Test: {len(test_files)} ảnh ({100*test_size:.0f}%)")
    print("─" * 60)
    
    return (train_files, train_labels), (test_files, test_labels)


def load_and_process_image(image_path, target_size=(224, 224)):
    """
    Load ảnh, resize, normalize
    """
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Normalize [0, 1]
    return img_array


def create_data_generator(mode='train'):
    """
    Tạo data augmentation cho train/test
    
    MODE TRAIN: Tăng cường (xoay, zoom, flip...)
    MODE TEST: Chỉ normalize, không tăng cường
    """
    if mode == 'train':
        # Data Augmentation: tạo sự đa dạng
        return ImageDataGenerator(
            rotation_range=20,           # Xoay ±20 độ
            width_shift_range=0.2,       # Dịch ngang 20%
            height_shift_range=0.2,      # Dịch dọc 20%
            horizontal_flip=True,        # Lật ngang
            zoom_range=0.2,              # Zoom ±20%
            shear_range=0.2,             # Cắt ±20%
            brightness_range=[0.8, 1.2], # Thay độ sáng
            fill_mode='nearest'
        )
    else:
        # Chỉ rescale (normalize)
        return ImageDataGenerator(rescale=1./255.)


# ═══════════════════════════════════════════════════════════════════
# 2. XÂY DỰNG MODEL - TRANSFER LEARNING
# ═══════════════════════════════════════════════════════════════════

def build_transfer_learning_model(input_shape=(224, 224, 3), learning_rate=0.001):
    """
    Xây dựng model sử dụng Transfer Learning (MobileNetV2)
    
    Cấu trúc:
    - Input: ảnh 224x224
    - MobileNetV2 (pre-trained ImageNet): chiết xuất features
    - Global Average Pooling: giảm chiều
    - Dense layers + Dropout: phân loại
    - Output: sigmoid để được xác suất [0, 1]
    """
    
    # Load MobileNetV2 đã train trên ImageNet
    base_model = MobileNetV2(
        input_shape=input_shape,
        include_top=False,  # Bỏ layer phân loại ImageNet
        weights='imagenet'
    )
    
    # Đóng bang các layer của base model (fine-tuning)
    # Chỉ train lại layer cuối cùng
    base_model.trainable = False
    
    print("🧠 Transfer Learning với MobileNetV2:")
    print(f"   - Pre-trained trên ImageNet (1.2M ảnh)")
    print(f"   - Base layers: {len(base_model.layers)} layers")
    print(f"   - Đóng bang base model, train layer cuối")
    print("─" * 60)
    
    # Xây dựng model tuỳ chỉnh
    model = models.Sequential([
        # Input
        layers.Input(shape=input_shape),
        
        # Pre-trained backbone
        base_model,
        
        # Chiết xuất features toàn cục
        layers.GlobalAveragePooling2D(),
        
        # Fully connected layers
        layers.Dense(256, activation='relu', 
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(128, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Output: 2 classes (ripe=1, unripe=0)
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',  # Lỗi cho 2 lớp
        metrics=['accuracy', 
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')]
    )
    
    return model


# ═══════════════════════════════════════════════════════════════════
# 3. TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_model(model, train_files, train_labels, test_files, test_labels,
                batch_size=16, epochs=20, val_split=0.2):
    """
    Huấn luyện model
    """
    # Chuẩn bị dữ liệu
    X_train = np.array([load_and_process_image(f) for f in train_files])
    y_train = np.array(train_labels)
    
    X_test = np.array([load_and_process_image(f) for f in test_files])
    y_test = np.array(test_labels)
    
    print(f"📦 Data shapes:")
    print(f"   - X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"   - X_test: {X_test.shape}, y_test: {y_test.shape}")
    print("─" * 60)
    
    # Data augmentation cho training
    train_gen = create_data_generator(mode='train')
    
    # Early stopping: dừng khi validation loss không cải thiện 3 epochs
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Model checkpoint: lưu best model
    os.makedirs('./models', exist_ok=True)
    checkpoint = keras.callbacks.ModelCheckpoint(
        CONFIG['model_save_path'],
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    # Training
    print("🚂 Bắt đầu training...")
    print("─" * 60)
    
    history = model.fit(
        train_gen.flow(X_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_split=val_split,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    return history, X_test, y_test


# ═══════════════════════════════════════════════════════════════════
# 4. EVALUATION & VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_model(model, X_test, y_test):
    """
    Đánh giá mô hình trên test set
    """
    print("\n" + "═" * 60)
    print("📊 ĐÁNH GIÁ MÔ HÌNH")
    print("═" * 60)
    
    # Dự đoán
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    
    # Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n📈 Kết quả trên Test Set ({len(y_test)} ảnh):")
    print(f"   - Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"   - Precision: {prec:.4f} (chính xác khi dự đoán ripe)")
    print(f"   - Recall:    {rec:.4f} (tìm được bao nhiêu trái ripe)")
    print(f"   - F1-Score:  {f1:.4f} (trung bình của precision & recall)")
    print("─" * 60)
    
    # Classification report
    print("\n📋 Chi tiết từng lớp:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Unripe (0)', 'Ripe (1)']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n🔍 Confusion Matrix:")
    print(f"   ┌─────────────────┬──────────────┬──────────────┐")
    print(f"   │ Actual \\ Predict│   Unripe     │    Ripe      │")
    print(f"   ├─────────────────┼──────────────┼──────────────┤")
    print(f"   │     Unripe      │     {cm[0,0]:3d}      │     {cm[0,1]:3d}      │")
    print(f"   │      Ripe       │     {cm[1,0]:3d}      │     {cm[1,1]:3d}      │")
    print(f"   └─────────────────┴──────────────┴──────────────┘")
    
    return y_pred, y_pred_proba


def plot_training_history(history):
    """
    Vẽ đồ thị training history
    """
    os.makedirs('./plots', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss curve
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curve
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(CONFIG['history_plot_path'], dpi=150)
    print(f"\n📊 Đã lưu đồ thị: {CONFIG['history_plot_path']}")
    plt.show()


def plot_confusion_matrix(y_test, y_pred):
    """
    Vẽ confusion matrix
    """
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Unripe', 'Ripe'],
               yticklabels=['Unripe', 'Ripe'],
               ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix - Test Set', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('./plots/confusion_matrix.png', dpi=150)
    print(f"📊 Đã lưu confusion matrix: ./plots/confusion_matrix.png")
    plt.show()


# ═══════════════════════════════════════════════════════════════════
# 5. MAIN - CHẠY TRAINING
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🍓 STRAWBERRY RIPENESS CLASSIFICATION - TRAINING")
    print("═" * 60)
    
    # 1. Load dữ liệu
    (train_files, train_labels), (test_files, test_labels) = load_and_split_data(
        CONFIG['ripe_dir'],
        CONFIG['unripe_dir'],
        test_size=CONFIG['test_size']
    )
    
    # 2. Xây dựng model
    model = build_transfer_learning_model(
        input_shape=(*CONFIG['img_size'], 3),
        learning_rate=CONFIG['learning_rate']
    )
    
    print("\n🏗️  Model Architecture:")
    model.summary()
    print("─" * 60)
    
    # 3. Training
    history, X_test, y_test = train_model(
        model,
        train_files, train_labels,
        test_files, test_labels,
        batch_size=CONFIG['batch_size'],
        epochs=CONFIG['epochs'],
        val_split=CONFIG['val_split']
    )
    
    # 4. Evaluation
    y_pred, y_pred_proba = evaluate_model(model, X_test, y_test)
    
    # 5. Visualization
    plot_training_history(history)
    plot_confusion_matrix(y_test, y_pred)
    
    print("\n" + "═" * 60)
    print("✅ Training hoàn thành!")
    print(f"💾 Model đã lưu: {CONFIG['model_save_path']}")
    print("═" * 60)
