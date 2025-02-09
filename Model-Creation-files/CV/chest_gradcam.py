

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns


data_dir = {
    "healthy": "/root/.cache/kagglehub/datasets/pranavraikokte/covid19-image-dataset/versions/2/Covid19-dataset/train/Normal",
    "covid-19": "/root/.cache/kagglehub/datasets/pranavraikokte/covid19-image-dataset/versions/2/Covid19-dataset/train/Covid",
    "pneumonia": "/root/.cache/kagglehub/datasets/pranavraikokte/covid19-image-dataset/versions/2/Covid19-dataset/train/Viral Pneumonia",
    "tuberculosis": "/content/Tuberculosis"
}

IMG_SIZE = (128, 128)
LIMIT = 70
CLASSES = list(data_dir.keys())


def load_images(data_dir, limit=LIMIT):
    images, labels = [], []
    for label, path in data_dir.items():
        count = 0
        for file in os.listdir(path):
            if count >= limit:
                break
            try:
                img = load_img(os.path.join(path, file), target_size=IMG_SIZE)
                img = img_to_array(img) / 255.0  # Normalize
                images.append(img)
                labels.append(CLASSES.index(label))
                count += 1
            except Exception as e:
                print(f"Error loading image {file}: {e}")
    return np.array(images), np.array(labels)


images, labels = load_images(data_dir)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42, stratify=labels)

# encoding
y_train = np.eye(len(CLASSES))[y_train]
y_test = np.eye(len(CLASSES))[y_test]

# Step 1: Build the Model
def build_model(input_shape=(128, 128, 3), num_classes=4):
    model = Sequential([
      Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
      MaxPooling2D(pool_size=(2, 2)),
      Conv2D(64, (3, 3), activation='relu'),
      MaxPooling2D(pool_size=(2, 2)),
      Flatten(),
      Dense(128, activation='relu'),
      Dropout(0.5),
      Dense(len(CLASSES), activation='softmax')
    ])
    return model

# Step 2: Compile the Model
chest_model = build_model()
chest_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = chest_model.fit(X_train, y_train, epochs=15, validation_split=0.2, batch_size=16)
print(chest_model.summary())

test_loss, test_acc = chest_model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc}")

# training accuracy and vali accuracy ploting
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

plt.show()

# Predictions on test data
y_pred = chest_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=CLASSES, yticklabels=CLASSES, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Classification Report
print("Classification Report:")
print(classification_report(y_test_classes, y_pred_classes, target_names=CLASSES))

# Testing with new images
def predict_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = chest_model.predict(img_array)
    predicted_class = CLASSES[np.argmax(prediction)]
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

# test_image = "/root/.cache/kagglehub/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone/versions/1/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Stone/Stone- (541).jpg"
# predict_image(test_image)

image_paths = [
    "/root/.cache/kagglehub/datasets/pranavraikokte/covid19-image-dataset/versions/2/Covid19-dataset/train/Viral Pneumonia/046.jpeg",
    "/root/.cache/kagglehub/datasets/pranavraikokte/covid19-image-dataset/versions/2/Covid19-dataset/train/Covid/07.jpg",
    "/root/.cache/kagglehub/datasets/pranavraikokte/covid19-image-dataset/versions/2/Covid19-dataset/train/Normal/059.jpeg",
    "/content/Tuberculosis/TEST_px33.jpg"
]

for image_path in image_paths:
    predict_image(image_path)

def plot_predictions(X_test, y_test, y_pred):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.ravel()
    for i in range(9):
        ax = axes[i]
        ax.imshow(X_test[i])
        ax.axis('off')
        ax.set_title(f"True: {CLASSES[np.argmax(y_test[i])]} | Pred: {CLASSES[np.argmax(y_pred[i])]}")
    plt.suptitle("True vs Predicted")
    plt.show()

plot_predictions(X_test, y_test, y_pred)

import tensorflow as tf
from tensorflow.keras.models import Model
import cv2

# Simulate Model Initialization
_ = chest_model.predict(np.zeros((1, 128, 128, 3)))  # Initialize model with dummy data

# Step 3: Grad-CAM Functions
def compute_gradcam(model, img_array, layer_name, class_idx):


    grad_model = Model([model.inputs], [model.get_layer("conv2d_1").output, model.get_layer("dense_1").output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Global average pooling

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(conv_outputs * pooled_grads, axis=-1)
    heatmap = np.maximum(heatmap, 0)  # ReLU to remove negative values
    heatmap /= np.max(heatmap)  # Normalize to [0, 1]
    return heatmap

def overlay_heatmap(img, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    overlayed_image = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
    return overlayed_image

def gradcam_visualization(model, image_path, layer_name='conv2d_1'):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict class
    predictions = model.predict(img_array)
    class_idx = np.argmax(predictions)
    class_label = ["healthy", "covid-19", "pneumonia", "tuberculosis"][class_idx]  # Replace with actual class names

    # Compute Grad-CAM
    heatmap = compute_gradcam(model, img_array, layer_name, class_idx)

    # Load original image
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (128, 128))

    # Overlay heatmap
    overlayed_img = overlay_heatmap(original_img, heatmap)

    # Plot Results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='viridis')
    plt.title("Grad-CAM Heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlayed_img)
    plt.title(f"Overlayed - Predicted: {class_label}")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# # Step 4: Test Grad-CAM
# test_image = "/root/.cache/kagglehub/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone/versions/1/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Tumor/Tumor- (1493).jpg"
# gradcam_visualization(chest_model, test_image, layer_name='conv2d_1')

image_paths = [
    "/root/.cache/kagglehub/datasets/pranavraikokte/covid19-image-dataset/versions/2/Covid19-dataset/train/Viral Pneumonia/046.jpeg",
    "/root/.cache/kagglehub/datasets/pranavraikokte/covid19-image-dataset/versions/2/Covid19-dataset/train/Covid/07.jpg",
    "/root/.cache/kagglehub/datasets/pranavraikokte/covid19-image-dataset/versions/2/Covid19-dataset/train/Normal/059.jpeg",
    "/content/Tuberculosis/TEST_px33.jpg"
]

for image_path in image_paths:
    gradcam_visualization(chest_model, image_path, layer_name='conv2d_1')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def saliency_map(model, img_path, target_size=(128, 128), class_index=None):
    """
    Generate and display the saliency map of the input image.

    Parameters:
    - model: Trained Keras model.
    - img_path: Path to the input image.
    - target_size: Target image size to resize to (default: (128, 128)).
    - class_index: Target class index for which saliency is computed.
    """
    # Load and preprocess the image
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_tensor = tf.convert_to_tensor(np.expand_dims(img_array, axis=0), dtype=tf.float32)  # Ensure tf.Tensor

    # Ensure gradients are tracked
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        if class_index is None:
            class_index = np.argmax(predictions[0])
        print(f"Predicted class: {class_index}")

        # Target output for the chosen class
        target_output = predictions[:, class_index]

    # Calculate gradients
    grads = tape.gradient(target_output, img_tensor)
    grads_abs = tf.abs(grads)  # Use absolute value of gradients
    saliency = np.max(grads_abs, axis=-1)[0]  # Reduce across color channels

    # Visualize the saliency map
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Original Image")
    plt.axis('off')

    # Saliency Map
    plt.subplot(1, 2, 2)
    plt.imshow(saliency, cmap='jet')
    plt.title(f"Saliency Map (Class: {class_index})")
    plt.axis('off')

    plt.show()

# test_image_path = "/root/.cache/kagglehub/datasets/nazmul0087/ct-kidney-dataset-normal-cyst-tumor-and-stone/versions/1/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/Tumor/Tumor- (1493).jpg"
# saliency_map(chest_model, test_image_path)

image_paths = [
    "/root/.cache/kagglehub/datasets/pranavraikokte/covid19-image-dataset/versions/2/Covid19-dataset/train/Viral Pneumonia/046.jpeg",
    "/root/.cache/kagglehub/datasets/pranavraikokte/covid19-image-dataset/versions/2/Covid19-dataset/train/Covid/07.jpg",
    "/root/.cache/kagglehub/datasets/pranavraikokte/covid19-image-dataset/versions/2/Covid19-dataset/train/Normal/059.jpeg",
    "/content/Tuberculosis/TEST_px33.jpg"
]

for image_path in image_paths:
    saliency_map(chest_model, image_path)

# Save the trained model
model_save_path = 'chest_disease_classification_model.h5'
chest_model.save(model_save_path)
print(f"Model saved to {model_save_path}")

