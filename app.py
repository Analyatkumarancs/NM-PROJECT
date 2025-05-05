import os
import cv2
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ✅ Define dataset paths
train_dir = 'modified-dataset/train'
val_dir = 'modified-dataset/val'

# ✅ Verify directories exist
if not os.path.isdir(train_dir):
    raise FileNotFoundError(f"❌ Training directory does not exist: {train_dir}")
if not os.path.isdir(val_dir):
    raise FileNotFoundError(f"❌ Validation directory does not exist: {val_dir}")
print("✅ Dataset directories verified.")

# ✅ OPTIONAL: Preview 5 sample images using OpenCV
sample_class_dir = os.path.join(train_dir, os.listdir(train_dir)[1])  # Pick the first class folder
image_files = [os.path.join(sample_class_dir, f) for f in os.listdir(sample_class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Showing 5 sample images from: {sample_class_dir}")
for i, img_path in enumerate(image_files[:5]):
    img = cv2.imread(img_path)  # Read image using OpenCV
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
    plt.subplot(1, 5, i+1)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"Image {i+1}")

plt.tight_layout()
plt.show()

# ✅ Image data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

# ✅ Load image datasets
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# ✅ Build the model using ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False  # Freeze base model

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')  # Output layer for multi-class
])

# ✅ Compile the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Add early stopping to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ✅ Train the model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=[early_stopping]
)

# ✅ Evaluate on validation data
loss, accuracy = model.evaluate(val_data)
print(f"\n✅ Model Accuracy on Validation Set: {accuracy * 100:.2f}%")

# ✅ Plot training vs. validation accuracy
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# ✅ Plot training vs. validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()