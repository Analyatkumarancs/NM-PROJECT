import os
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import json

# ✅ Verify dataset paths
train_dir = 'modified-dataset/train'
val_dir = 'modified-dataset/val'

if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
    raise FileNotFoundError("❌ One or both dataset directories are missing.")

print("✅ Dataset directories verified.")

# ✅ Preview 5 sample images
sample_class_dir = os.path.join(train_dir, os.listdir(train_dir)[0])
image_files = [os.path.join(sample_class_dir, f)
               for f in os.listdir(sample_class_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Showing 5 sample images from: {sample_class_dir}")
plt.figure(figsize=(12, 3))
for i, img_path in enumerate(image_files[:5]):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.subplot(1, 5, i+1)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.title(f"Image {i+1}")
plt.tight_layout()
plt.show()

# ✅ Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

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

# ✅ Save class indices
with open("class_indices.json", "w") as f:
    json.dump(train_data.class_indices, f)
print("✅ Class indices saved to class_indices.json")

# ✅ Build ResNet50 model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dense(train_data.num_classes, activation='softmax')
])

model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# ✅ Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# ✅ Train model
history = model.fit(
    train_data,
    epochs=10,
    validation_data=val_data,
    callbacks=[early_stopping]
)

# ✅ Evaluate model
loss, accuracy = model.evaluate(val_data)
print(f"\n✅ Final Validation Accuracy: {accuracy * 100:.2f}%")

# ✅ Save model
model.save("e_waste_classifier_resnet50.h5")
print("✅ Model saved to e_waste_classifier_resnet50.h5")

# ✅ Plot accuracy & loss
plt.figure(figsize=(12, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
