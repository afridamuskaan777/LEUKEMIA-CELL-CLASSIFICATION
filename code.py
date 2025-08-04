import numpy as np
import pandas as pd
import os
import cv2 as cv
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Define Directories
base_dir = "/kaggle/input/leukemia-classification/C-NMC_Leukemia/"
train_folds = [
    os.path.join(base_dir, "training_data/fold_0"),
    os.path.join(base_dir, "training_data/fold_1"),
    os.path.join(base_dir, "training_data/fold_2"),
]

all_dirs = [os.path.join(fold, "all") for fold in train_folds]
hem_dirs = [os.path.join(fold, "hem") for fold in train_folds]

# Get Image Paths and Labels
def get_image_paths(folder):
    return [os.path.join(folder, fname) for fname in os.listdir(folder)]

img_data = []
for folder in all_dirs + hem_dirs:
    img_data.extend(get_image_paths(folder))

data = pd.DataFrame({"img_data": img_data, "labels": [np.nan for _ in range(len(img_data))]})
data.loc[0:7271, "labels"] = 1  # ALL
data.loc[7272:10660, "labels"] = 0  # HEM
data["labels"] = data["labels"].astype("int64")

# Preprocessing and Loading Images
img_list = []
for path in data["img_data"]:
    image = cv.imread(path)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    result = cv.bitwise_and(image, image, mask=thresh)
    result[thresh == 0] = [255, 255, 255]
    
    x, y, _ = np.where(result > 0)
    mnx, mxx = np.min(x), np.max(x)
    mny, mxy = np.min(y), np.max(y)
    crop_img = image[mnx:mxx, mny:mxy, :]
    crop_img_resized = cv.resize(crop_img, (224, 224))
    img_list.append(crop_img_resized)

X = np.array(img_list)
y = np.array(data["labels"])

# This is the visualization block that was interrupted in the original script
img_list_visualize = []
for path in data["img_data"]:
    image = cv.imread(path)
    
    # Display the original image
    plt.figure(figsize=(6,6))
    plt.subplot(1, 3, 1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Display grayscale image
    plt.subplot(1, 3, 2)
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.axis('off')
    
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Display CLAHE enhanced grayscale image
    plt.subplot(1, 3, 3)
    plt.imshow(gray, cmap='gray')
    plt.title("CLAHE Enhanced")
    plt.axis('off')
    
    plt.show()

    thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)
    
    # Display thresholded image
    plt.figure(figsize=(6,6))
    plt.imshow(thresh, cmap='gray')
    plt.title("Thresholded Image")
    plt.axis('off')
    plt.show()

    result = cv.bitwise_and(image, image, mask=thresh)
    result[thresh == 0] = [255, 255, 255]
    
    # Show the result after applying mask
    plt.figure(figsize=(6,6))
    plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    plt.title("Result after Bitwise AND")
    plt.axis('off')
    plt.show()
    
    x, y, _ = np.where(result > 0)
    mnx, mxx = np.min(x), np.max(x)
    mny, mxy = np.min(y), np.max(y)
    crop_img = image[mnx:mxx, mny:mxy, :]
    crop_img_resized = cv.resize(crop_img, (224, 224))

    img_list_visualize.append(crop_img_resized)

# Re-assign X and y in case the visualization loop was intended to be the main one
# X = np.array(img_list_visualize)
# y = np.array(data["labels"])


# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocess Images for ResNet50
X_train = preprocess_input(X_train)
X_test = preprocess_input(X_test)

# Build and Compile the Model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

# Callbacks
checkpoint = ModelCheckpoint("resnet50_leukemia_model.keras", monitor="val_accuracy", save_best_only=True, verbose=1)
early_stopping = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

# Train the Model
history = model.fit(X_train, y_train,
                    validation_split=0.2,
                    epochs=25,
                    batch_size=32,
                    callbacks=[checkpoint, early_stopping],
                    verbose=1)

# Evaluate the Model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Visualize Training History
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

model.save("resnet50_leukemia_final_model.h5")
