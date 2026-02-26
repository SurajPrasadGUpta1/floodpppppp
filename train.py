import os
import cv2
import numpy as np
import tensorflow as tf
import rasterio
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

IMG_SIZE = 256
DATASET_PATH = "dataset_sen1"
EPOCHS = 30
BATCH_SIZE = 4

# ---------------- LOAD DATA ----------------
def load_data():
    images = []
    masks = []

    img_dir = os.path.join(DATASET_PATH, "s1")
    mask_dir = os.path.join(DATASET_PATH, "labels")

    print("Loading dataset...")

    files = [f for f in os.listdir(img_dir) if f.endswith("_S1Hand.tif")]
    print("Total TIFF files found:", len(files))

    for file in files:
        img_path = os.path.join(img_dir, file)

        mask_name = file.replace("_S1Hand.tif", "_LabelHand.tif")
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            continue

        try:
            with rasterio.open(img_path) as src:
                img = src.read(1).astype(np.float32)

            with rasterio.open(mask_path) as src:
                mask = src.read(1).astype(np.float32)
        except:
            continue

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        mask = (mask > 0).astype(np.float32)

        img = np.expand_dims(img, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# ---------------- U-NET MODEL ----------------
def unet_model():
    inputs = layers.Input((IMG_SIZE, IMG_SIZE, 1))

    c1 = layers.Conv2D(64, 3, activation="relu", padding="same")(inputs)
    c1 = layers.Conv2D(64, 3, activation="relu", padding="same")(c1)
    p1 = layers.MaxPooling2D()(c1)

    c2 = layers.Conv2D(128, 3, activation="relu", padding="same")(p1)
    c2 = layers.Conv2D(128, 3, activation="relu", padding="same")(c2)
    p2 = layers.MaxPooling2D()(c2)

    c3 = layers.Conv2D(256, 3, activation="relu", padding="same")(p2)
    c3 = layers.Conv2D(256, 3, activation="relu", padding="same")(c3)

    u1 = layers.UpSampling2D()(c3)
    u1 = layers.concatenate([u1, c2])
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(u1)
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same")(c4)

    u2 = layers.UpSampling2D()(c4)
    u2 = layers.concatenate([u2, c1])
    c5 = layers.Conv2D(64, 3, activation="relu", padding="same")(u2)
    c5 = layers.Conv2D(64, 3, activation="relu", padding="same")(c5)

    outputs = layers.Conv2D(1, 1, activation="sigmoid")(c5)

    return models.Model(inputs, outputs)

# ---------------- TRAIN ----------------
X, y = load_data()
print("Dataset shape:", X.shape)

if X.shape[0] == 0:
    raise ValueError("❌ No images loaded. Check dataset paths!")

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = unet_model()
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

os.makedirs("model", exist_ok=True)
model.save("model/flood_unet_sen1.h5")

print("✅ Sen1Floods11 U-Net training completed and model saved!")
