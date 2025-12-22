
MODEL BUILDING Unet

Import and prepare file paths
"""

# Commented out IPython magic to ensure Python compatibility.
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
# %matplotlib inline

from google.colab import drive
drive.mount('/content/drive')

"""Loading the Ground and Mask Images"""

import cv2
import glob
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers


# === Helper to extract index from filename ===
def extract_index(path, prefix):
    basename = os.path.basename(path)
    number = basename.replace(prefix, '').replace('.png', '')
    return int(number)

# === Step 1: Load, sort numerically, and filter paths ===
image_paths = glob.glob("/content/drive/MyDrive/large/imag/*.png")
mask_paths = glob.glob("/content/drive/MyDrive/large/mas/*.png")

# Sort numerically based on the image/mask number
image_paths = sorted(image_paths, key=lambda p: extract_index(p, 'image'))
mask_paths = sorted(mask_paths, key=lambda p: extract_index(p, 'mask'))

# Filter for image132 to image656 only
image_paths = [p for p in image_paths if 1 <= extract_index(p, 'image') <= 1671]
mask_paths = [p for p in mask_paths if 1 <= extract_index(p, 'mask') <= 1671]

# === Step 2: Resize and preprocess ===
input_images = []
input_masks = []

for img_path, mask_path in zip(image_paths, mask_paths):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_images.append(img)

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
    input_masks.append(mask)

input_images = np.array(input_images, dtype=np.float32) / 255.0
input_masks = np.array(input_masks, dtype=np.int32)[..., np.newaxis]
# === Train-validation split (90% training, 10% validation) ===
X_train, X_val, y_train, y_val = train_test_split(
    input_images, input_masks, test_size=0.1, random_state=42
)

input_images.shape

input_masks.shape

"""Space utilization"""

import os
import cv2
import numpy as np
import random
import tensorflow as tf

def data_generator(image_list, mask_list, batch_size=8, augment=True, target_size=(256, 256)):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    while True:
        zipped = list(zip(image_list, mask_list))
        random.shuffle(zipped)
        for i in range(0, len(zipped), batch_size):
            batch_images = []
            batch_masks = []

            for img_path, mask_path in zipped[i:i + batch_size]:
                # Load and preprocess image
                img = cv2.imread(img_path)
                img = cv2.resize(img, target_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img = (img - mean) / std

                # Load and preprocess mask
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
                mask = (mask > 127).astype(np.float32)

                # Augmentation
                if augment:
                    if random.random() > 0.5:
                        img = np.fliplr(img)
                        mask = np.fliplr(mask)
                    if random.random() > 0.5:
                        img = np.flipud(img)
                        mask = np.flipud(mask)
                    if random.random() > 0.5:
                        angle = random.randint(-15, 15)
                        matrix = cv2.getRotationMatrix2D((target_size[1] // 2, target_size[0] // 2), angle, 1)
                        img = cv2.warpAffine(img, matrix, target_size)
                        mask = cv2.warpAffine(mask, matrix, target_size)

                # Ensure correct shape
                mask = np.expand_dims(mask, axis=-1)
                if img.shape == (256, 256, 3) and mask.shape == (256, 256, 1):
                    batch_images.append(img)
                    batch_masks.append(mask)

            if batch_images and batch_masks:
                yield np.array(batch_images, dtype=np.float32), np.array(batch_masks, dtype=np.float32)

# === Split paths into train/val ===
train_image_paths = image_paths[:int(len(image_paths) * 0.9)]
train_mask_paths = mask_paths[:int(len(mask_paths) * 0.9)]
val_image_paths = image_paths[int(len(image_paths) * 0.9):]
val_mask_paths = mask_paths[int(len(mask_paths) * 0.9):]


# === Dataset parameters ===
BATCH_SIZE = 16

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(train_image_paths, train_mask_paths, BATCH_SIZE, augment=True),
    output_signature=(
        tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32),
    )
)

val_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(val_image_paths, val_mask_paths, BATCH_SIZE, augment=False),
    output_signature=(
        tf.TensorSpec(shape=(None, 256, 256, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, 256, 256, 1), dtype=tf.float32),
    )
)

# === Steps for training
steps_per_epoch = len(train_image_paths) // BATCH_SIZE
validation_steps = len(val_image_paths) // BATCH_SIZE

sample_mask = cv2.imread(mask_paths[0], cv2.IMREAD_GRAYSCALE)
sample_mask = cv2.resize(sample_mask, (256, 256), interpolation=cv2.INTER_NEAREST)
print("Unique values in sample mask:", np.unique(sample_mask))

"""Defining Unet Model with ResNet50 Encoder"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50

def conv_block(x, filters, dropout_rate=0.0):
    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    return x

def build_unet_resnet50(input_shape=(256, 256, 3), output_channels=1):
    inputs = tf.keras.Input(shape=input_shape)

    # Load ResNet50 as encoder
    base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=inputs)

    # Extract relevant feature maps
    skips = [
        base_model.get_layer("conv1_relu").output,     # 128x128
        base_model.get_layer("conv2_block3_out").output, # 64x64
        base_model.get_layer("conv3_block4_out").output, # 32x32
        base_model.get_layer("conv4_block6_out").output, # 16x16
    ]
    bottleneck = base_model.get_layer("conv5_block3_out").output  # 8x8

    # Decoder / Upsampling path
    x = bottleneck
    for skip, filters in zip(reversed(skips), [512, 256, 128, 64]):
        x = layers.UpSampling2D()(x)
        x = layers.Concatenate()([x, skip])
        x = conv_block(x, filters)

    # Final upsampling to match input resolution
    x = layers.UpSampling2D()(x)
    x = conv_block(x, 32)

    outputs = layers.Conv2D(output_channels, 1, activation='sigmoid')(x)

    return Model(inputs, outputs, name="ResNet50-U-Net")

"""Compile and Training the Model"""

from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# === Custom IoU metric ===
def iou_metric(y_true, y_pred, smooth=1e-7):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1,2,3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1,2,3]) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return tf.reduce_mean(iou)


# === Dice Loss ===
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# === Weighted Binary Cross-Entropy
def weighted_bce(y_true, y_pred, weight=5.0):
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.where(tf.equal(y_true, 1), bce * weight, bce)

# === Combo Loss: Weighted BCE + Dice
def combo_loss(y_true, y_pred, alpha=0.5, weight=5.0):
    bce = weighted_bce(y_true, y_pred, weight)
    d_loss = dice_loss(y_true, y_pred)
    return alpha * bce + (1 - alpha) * d_loss

# === Build U-Net Model
model = build_unet_resnet50(output_channels=1)

# === Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# === Compile the model
model.compile(
    optimizer=optimizer,
    loss=combo_loss,
    metrics=['accuracy', iou_metric]
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_iou_metric',
        mode='max',  # <--- this line is essential
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_iou_metric',
        mode='max',  # <--- also update here if you're tracking val_iou_metric
        save_best_only=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
]
# === Training setup

EPOCHS = 50


# === Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=callbacks
)

"""Save Model"""

model.save("v2_1671final_model.keras")

from google.colab import files
files.download("v2_1671final_model.keras")

"""Display Results"""

import matplotlib.pyplot as plt
import cv2
import numpy as np

# === Postprocessing Function ===
def postprocess_mask(mask):
    mask = (mask * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
    return cleaned

# === Predict, Postprocess, and Show Up to 3 Samples Safely ===
X_batch, y_true_batch = next(iter(val_dataset.take(1)))  # Ensure proper batch extraction

y_pred_batch = model.predict(X_batch)
y_pred_batch = (y_pred_batch > 0.5).astype(np.float32)

num_samples = min(3, X_batch.shape[0])

plt.figure(figsize=(15, num_samples * 4))
for i in range(num_samples):
    image = X_batch[i].numpy() if hasattr(X_batch[i], 'numpy') else X_batch[i]
    true_mask = y_true_batch[i].numpy().squeeze() if hasattr(y_true_batch[i], 'numpy') else y_true_batch[i].squeeze()
    raw_pred = y_pred_batch[i].squeeze()
    cleaned_pred = postprocess_mask(raw_pred)

    plt.subplot(num_samples, 4, i * 4 + 1)
    plt.imshow((image * 0.229 + 0.485))  # Denormalize roughly for visualization
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(num_samples, 4, i * 4 + 2)
    plt.imshow(true_mask, cmap='gray')
    plt.title("True Mask")
    plt.axis("off")

    plt.subplot(num_samples, 4, i * 4 + 3)
    plt.imshow(raw_pred, cmap='gray')
    plt.title("Predicted Mask (Raw)")
    plt.axis("off")

    plt.subplot(num_samples, 4, i * 4 + 4)
    plt.imshow(cleaned_pred, cmap='gray')
    plt.title("Predicted Mask (Cleaned)")
    plt.axis("off")

plt.tight_layout()
plt.show()

from google.colab import files

# Save as .keras (recommended by Keras 3.x)
model.save("unet_model.keras")

# Download the .keras file
files.download("unet_model.keras")

import json
from google.colab import files

# Handle both Keras History object and plain dict
history_dict = history.history if hasattr(history, 'history') else history

# Save training history to JSON file
with open('history.json', 'w') as f:
    json.dump(history_dict, f)

# Download it
files.download('history.json')

model.save('full_model.h5')
files.download('full_model.h5')

import numpy as np
import matplotlib.pyplot as plt
import random

# Set how many samples you want to visualize
num_samples = 5

# Get a batch from the test dataset
X_batch, y_true_batch = next(iter(test_dataset))

# Predict the masks
y_pred_batch = model.predict(X_batch)
y_pred_batch = (y_pred_batch > 0.5).astype(np.uint8)

# Randomly select sample indices from the batch
indices = random.sample(range(len(X_batch)), k=min(num_samples, len(X_batch)))

# Plot input, true mask, and predicted mask
plt.figure(figsize=(15, num_samples * 3))

for i, idx in enumerate(indices):
    img = X_batch[idx]
    true_mask = y_true_batch[idx].squeeze()
    pred_mask = y_pred_batch[idx].squeeze()

    plt.subplot(num_samples, 3, i * 3 + 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(num_samples, 3, i * 3 + 2)
    plt.imshow(true_mask, cmap='gray')
    plt.title("True Mask")
    plt.axis("off")

    plt.subplot(num_samples, 3, i * 3 + 3)
    plt.imshow(pred_mask, cmap='gray')
    plt.title("Predicted Mask")
    plt.axis("off")

plt.tight_layout()
plt.show()

