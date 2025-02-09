import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
import pandas as pd
from tensorflow import keras


def predict_organ(image_path):
    organ_model = load_model('organ_classifier.h5')
    IMG_SIZE = (128,128)
    # print("identifying organ")
    CLASSES_ORGANS = ['Chest', 'Liver-Disease', 'Liver-Tumor', 'Brain', 'brain-dcm', 'Eye', 'Kidney']
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    # print(img_array)
    prediction = organ_model.predict(img_array)
    predicted_class = CLASSES_ORGANS[np.argmax(prediction)]
    # print(predicted_class)
    return predicted_class

def kidney_stone_model(image_path, filename, model_path="/content/runs/detect/kidney_stone.pt/weights/best.pt"):
 
    # Load the trained model
    model = YOLO(model_path)

    # Run inference on the image
    results = model.predict(source=image_path, save=False)

    # Load the image
    img = cv2.imread(image_path)

    # Tumor details
    kidney_details = []

    # Loop through results and process each bounding box
    for detection in results[0].boxes.data:  # Each detection
        # Extract bounding box coordinates and details
        x1, y1, x2, y2 = map(int, detection[:4])  # Bounding box coordinates
        confidence = float(detection[4])  # Confidence score
        label = int(detection[5])  # Class label


        kidney_label = " "

        if label == 0:
            kidney_label = "kidney"
        else:
            kidney_label = "stone"


        # Calculate size (area) of the tumor
        width = (x2 - x1) * 0.2645833333
        height = (y2 - y1) * 0.2645833333
        area = width * height

        # Get tumor shape and metrics
        kidney_area, kidney_perimeter, shape = carve_tumor_edges(img, x1, y1, x2, y2)

        # Determine severity based on tumor size
        if area < 1000:
            severity = "Mild"
        elif 1000 <= area < 5000:
            severity = "Moderate"
        else:
            severity = "Severe"

        # Determine tumor location (e.g., divide liver into quadrants)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        if center_x < img.shape[1] // 2 and center_y < img.shape[0] // 2:
            location = "Upper Left lobe"
        elif center_x >= img.shape[1] // 2 and center_y < img.shape[0] // 2:
            location = "Upper Right lobe"
        elif center_x < img.shape[1] // 2 and center_y >= img.shape[0] // 2:
            location = "Lower Left lobe"
        else:
            location = "Lower Right lobe"

        # Append details
        kidney_details.append({
            "label": kidney_label,
            "confidence": round(confidence, 2),
            "area": round(area, 2),
            "size": f"{width:.2f}mm x {height:.2f}mm",
            "location": location,
            "severity": severity,
            "shape": shape,
            "tumor_area": round(kidney_area, 2),
            "tumor_perimeter": round(kidney_perimeter, 2)
        })

        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{kidney_label}, {severity} ({confidence:.2f})"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the image with bounding boxes as 'output.jpg'
    output_path = os.path.join('output', filename)
    cv2.imwrite(output_path, img)

    # Visualize predictions
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("Predictions")
    plt.show()

    # print(f"Output image saved as {output_path}")
    # print("\nTumor Details:")
    # for i, tumor in enumerate(kidney_details, 1):
    #     print(f"Tumor {i}: {tumor}")
    # return (kidney_details)
    count="single"
    features=[]
    # print(f"Output image saved as {output_path}")
    # print("\nTumor Details:")
    for i, tumor in enumerate(kidney_details, 1):
        # print(f"Tumor {i}: {tumor}")
        # feature=[tumor["size"],tumor["location"],tumor["severity"]]
        feature=[f'stone {i}',tumor["area"],tumor["size"],tumor["shape"],tumor["location"],tumor["severity"]]
        features.append(feature)
        if i>1:
          count="multiple"
    features.append(count)
    return "stone", features

def kidney_model(image_path):
    kidney = load_model('kidney_disease_classification_model.h5')
    IMG_SIZE = (128,128)
    CLASSES = ['cyst', 'stone', 'tumor', 'healthy']
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = kidney.predict(img_array)
    predicted_class = CLASSES[np.argmax(prediction)]
    # img = cv2.imread(image_path)
    # output_path = "output.jpg"
    # cv2.imwrite(output_path, img)

    # # Visualize predictions
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img_rgb)
    # plt.axis("off")
    # title="Prections: "+predicted_class
    # plt.title(title)
    return predicted_class

def eye_model(image_path):
    eye = load_model('eye_disease_classification_model.h5')
    IMG_SIZE = (128,128)
    CLASSES = ["cataract",  "diabetic_retinopathy", "glaucoma", "normal"]
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = eye.predict(img_array)
    predicted_class = CLASSES[np.argmax(prediction)]
    # img = cv2.imread(image_path)
    # output_path = "output.jpg"
    # cv2.imwrite(output_path, img)

    # # Visualize predictions
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img_rgb)
    # plt.axis("off")
    # title="Prections: "+predicted_class
    # plt.title(title)
    return predicted_class

def chest_model(image_path):
    chest = load_model('chest_disease_classification_model.h5')
    IMG_SIZE = (128,128)
    CLASSES = ['healthy', 'covid-19', 'pneumonia', 'tuberculosis']
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = chest.predict(img_array)
    predicted_class = CLASSES[np.argmax(prediction)]
    # img = cv2.imread(image_path)
    # output_path = "output.jpg"
    # cv2.imwrite(output_path, img)

    # # Visualize predictions
    # img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.figure(figsize=(10, 10))
    # plt.imshow(img_rgb)
    # plt.axis("off")
    # title="Prections: "+predicted_class
    # plt.title(title)
    return predicted_class

def carve_tumor_edges(image, x1, y1, x2, y2):
    """
    Carves out the exact edges of the tumor within the bounding box.
    """
    PIXEL_TO_MM = 0.264
    # Crop the ROI from the bounding box
    roi = image[y1:y2, x1:x2]

    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred_roi = cv2.GaussianBlur(gray_roi, (5, 5), 0)

    # Apply Canny edge detection
    edges = cv2.Canny(blurred_roi, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the ROI for visualization
    contour_img = cv2.cvtColor(gray_roi, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

    # Calculate tumor shape metrics
    tumor_area_pixels = 0
    tumor_perimeter_pixels = 0
    shape = "Unknown"
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        tumor_area_pixels = cv2.contourArea(largest_contour)
        tumor_perimeter_pixels = cv2.arcLength(largest_contour, True)

        # Calculate shape characteristics
        x, y, w, h = cv2.boundingRect(largest_contour)
        aspect_ratio = float(w) / h
        circularity = 4 * np.pi * (tumor_area_pixels / (tumor_perimeter_pixels ** 2))

        if circularity > 0.8:
            shape = "circular"
        elif aspect_ratio > 1.2:
            shape = "elongated (horizontal)"
        elif aspect_ratio < 0.8:
            shape = "elongated (vertical)"
        else:
            shape = "irregular"

    # # Display the edges and contours
    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(edges, cmap='gray')
    # plt.title("Edges")
    # plt.axis("off")

    # plt.subplot(1, 2, 2)
    # plt.imshow(contour_img)
    # plt.title("Contours")
    # plt.axis("off")
    # plt.show()

    # Convert area and perimeter to mm
    tumor_area_mm = tumor_area_pixels * (PIXEL_TO_MM ** 2)
    tumor_perimeter_mm = tumor_perimeter_pixels * PIXEL_TO_MM

    return tumor_area_mm, tumor_perimeter_mm, shape

def brain_model(image_path, filename, model_path="/content/drive/MyDrive/BioBERT/Brain_Tumor_Classification_best.pt"):
    # Load the trained YOLOv8 model
    model = YOLO(model_path)

    # Run inference on the image
    results = model.predict(source=image_path, save=False, conf=0.60)

    # Load the image
    img = cv2.imread(image_path)

    # Define class names
    class_names = {
        0: "glioma tumor",
        1: "meningioma tumor",
        2: "pituitary tumor"
    }
    PIXEL_TO_MM = 0.264
    
    tumor_name=""
    diseases=[]
    features=[]
    i=1
    count="single"
    # Loop through detections
    for detection in results[0].boxes.data:  # Each detection
        # Extract bounding box coordinates and details
        x1, y1, x2, y2 = map(int, detection[:4])  # Bounding box coordinates
        confidence = float(detection[4])  # Confidence score
        label = int(detection[5])  # Class label

        # Calculate tumor size and position
        width_px = x2 - x1
        height_px = y2 - y1
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Convert dimensions to mm
        width_mm = width_px * PIXEL_TO_MM
        height_mm = height_px * PIXEL_TO_MM

        # Determine location in the image
        img_height, img_width = img.shape[:2]
        vertical_pos = "top" if center_y < img_height // 3 else "bottom" if center_y > 2 * img_height // 3 else "middle"
        horizontal_pos = "left" if center_x < img_width // 3 else "right" if center_x > 2 * img_width // 3 else "center"
        location = f"{vertical_pos} {horizontal_pos}"

        # Get tumor shape and metrics
        tumor_area_mm, tumor_perimeter_mm, shape = carve_tumor_edges(img, x1, y1, x2, y2)

        # Determine severity
        if tumor_area_mm < 500:
            severity = "Mild"
        elif tumor_area_mm < 2000:
            severity = "Moderate"
        else:
            severity = "Severe"

        # Print tumor details
        tumor_name = class_names.get(label, "Unknown")
        # print(f"Tumor Detected: {tumor_name}")
        # print(f"Confidence: {confidence:.2f}")
        # print(f"Size: {width_mm:.2f}mm x {height_mm:.2f}mm")
        # print(f"Location: {location}")
        # print(f"Shape: {shape}")
        # print(f"Tumor Area: {tumor_area_mm:.2f} mm²")
        # print(f"Tumor Perimeter: {tumor_perimeter_mm:.2f} mm")
        # print(f"Severity: {severity}")
        diseases.append(tumor_name)
        # feature=[f"{width_mm:.2f}mm x {height_mm:.2f}mm", location, severity]
        feature=[f'tumor {i}',f"{tumor_area_mm:.2f} mm²",f"{width_mm:.2f}mm x {height_mm:.2f}mm",shape,location,severity]
        features.append(feature)
       
        if i>1:
            count="multiple"
        i+=1
    
        # Draw bounding box and details on the image
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"{tumor_name} ({confidence:.2f})"
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    features.append(count)
    # Save and display the output image
    output_path = os.path.join('output', filename)
    cv2.imwrite(output_path, img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("Tumor Detection Results")
    # print(f"Output image saved as {output_path}")
    diseases=set(diseases)
    diseases=list(diseases)
    if len(diseases)==1:
        diseases=diseases[0]
    return diseases,features

def liver_tumor_model(image_path, filename, model_path="/content/drive/MyDrive/PICT Techfiesta/liver_tumor_segmentation_model.pt"):
    """
    Enhances the YOLO model predictions by calculating tumor size, location, severity, and count.
    """
    # Load the trained model
    model = YOLO(model_path)

    # Run inference on the image
    results = model.predict(source=image_path, save=False)

    # Load the image
    img = cv2.imread(image_path)

    # Tumor details
    tumor_details = []

    # Loop through results and process each bounding box
    for detection in results[0].boxes.data:  # Each detection
        # Extract bounding box coordinates and details
        x1, y1, x2, y2 = map(int, detection[:4])  # Bounding box coordinates
        confidence = float(detection[4])  # Confidence score
        label = int(detection[5])  # Class label

        if confidence < 0.5:
            continue

        # Tumor label
        tumor_label = "tumor"

        # Calculate size (area) of the tumor
        width = (x2 - x1) * 0.2645833333
        height = (y2 - y1) * 0.2645833333
        area = width * height

        # Get tumor shape and metrics
        tumor_area, tumor_perimeter, shape = carve_tumor_edges(img, x1, y1, x2, y2)

        # Determine severity based on tumor size
        if area < 1000:
            severity = "Mild"
        elif 1000 <= area < 5000:
            severity = "Moderate"
        else:
            severity = "Severe"

        # Determine tumor location (e.g., divide liver into quadrants)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        if center_x < img.shape[1] // 2 and center_y < img.shape[0] // 2:
            location = "Upper Left lobe"
        elif center_x >= img.shape[1] // 2 and center_y < img.shape[0] // 2:
            location = "Upper Right lobe"
        elif center_x < img.shape[1] // 2 and center_y >= img.shape[0] // 2:
            location = "Lower Left lobe"
        else:
            location = "Lower Right lobe"

        # Append details
        tumor_details.append({
            "label": tumor_label,
            "confidence": round(confidence, 2),
            "area": round(area, 2),
            "size": f"{width:.2f}mm x {height:.2f}mm",
            "location": location,
            "severity": severity,
            "shape": shape,
            "tumor_area": round(tumor_area, 2),
            "tumor_perimeter": round(tumor_perimeter, 2)
        })

        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"Class {label}, {severity} ({confidence:.2f})"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save the image with bounding boxes as 'output.jpg'
    output_path = os.path.join('output', filename)
    cv2.imwrite(output_path, img)

    # Visualize predictions
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis("off")
    title="Prections: "+ "Liver Tumor"
    plt.title(title)
    count="single"
    features=[]
    # print(f"Output image saved as {output_path}")
    # print("\nTumor Details:")
    for i, tumor in enumerate(tumor_details, 1):
        # print(f"Tumor {i}: {tumor}")
        feature=[f'tumor {i}',tumor["area"],tumor["size"],tumor["shape"],tumor["location"],tumor["severity"]]
        features.append(feature)
        # print("x")
        if i>1:
          count="multiple"
    features.append(count)
    return "Liver-Tumor", features

def liver_model(image_path, filename, model_path="/content/drive/MyDrive/PICT Techfiesta/liver_disease_detection.pt"):
    """
    Enhances the YOLO model predictions by calculating tumor size, location, severity, and count.
    """
    # Load the trained model
    model = YOLO(model_path)

    # Run inference on the image
    results = model.predict(source=image_path, save=False)  # Do not save in 'runs/detect/'

    # Extract bounding boxes, labels, and confidences
    boxes = results[0].boxes.xyxy  # Bounding box coordinates
    labels = results[0].boxes.cls  # Class labels
    confidences = results[0].boxes.conf  # Confidence scores

    # Load the image
    img = cv2.imread(image_path)

    # Tumor details
    tumor_details = []

    # Loop through results and process each bounding box
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])  # Convert box coordinates to integers
        label = int(labels[i])  # Convert label to integer
        confidence = confidences[i]  # Confidence score



        # label name
        if label == 0:
          label = "ballooning"
        elif label == 1:
          label = "fibrosis"
        elif label == 2:
          label = "inflammation"
        else :
          label = "steatosis"

        # Calculate size (area) of the tumor
        width = round((x2 - x1) * 0.2645833333)
        height = round((y2 - y1) * 0.2645833333)
        area = width * height

        # Determine severity based on tumor size
        if area < 1000:
            severity = "Mild"
        elif 1000 <= area < 5000:
            severity = "Moderate"
        else:
            severity = "Severe"

        # Determine tumor location (e.g., divide liver into quadrants)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        if center_x < img.shape[1] // 2 and center_y < img.shape[0] // 2:
            location = "Upper Left lobe"
        elif center_x >= img.shape[1] // 2 and center_y < img.shape[0] // 2:
            location = "Upper Right lobe"
        elif center_x < img.shape[1] // 2 and center_y >= img.shape[0] // 2:
            location = "Lower Left lobe"
        else:
            location = "Lower Right lobe"

        # Append details
        tumor_details.append({
            "label": label,
            "confidence": round(float(confidence), 2),
            "area": area,
            "size": f"{width}mm x {height}mm",
            "location": location,
            "severity": severity
        })

        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"Class {label}, {severity} ({confidence:.2f})"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save the image with bounding boxes as 'output1.jpg'
    output_path = os.path.join('output', filename)
    cv2.imwrite(output_path, img)
  

    # Visualize predictions
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis("off")
    title="Prections: "+tumor_details
    plt.title(title)

    # print(f"Output image saved as {output_path}")
    # print("\nDisease Details:")
    # for i, tumor in enumerate(tumor_details, 1):
    #     print(f"Disease {i}: {tumor}")
    # print(tumor_details)
    # return tumor_details
    count="single"
    features=[]
    diseases=[]
    # print(f"Output image saved as {output_path}")
    # print("\nTumor Details:")
    for i, tumor in enumerate(tumor_details, 1):
        # print(f"Tumor {i}: {tumor}")
        diseases.append(tumor['label'])
        feature=[f'tumor {i}',tumor["area"],tumor["size"],"none",tumor["location"],tumor["severity"]]
        features.append(feature)
        if i>1:
          count="multiple"
    features.append(count)
    diseases=set(diseases)
    diseases=list(diseases)
    if len(diseases)==1:
        diseases=diseases[0]
    return diseases, features

def predict_disease(organ, image_path, filename):
    """Predict disease based on organ type and image."""
    if organ == 'Chest':
        disease=chest_model(image_path)
    elif organ == 'Liver-Disease':
        disease, features=liver_model(image_path, filename, "liver_disease_detection.pt")
        # print(disease)
        # print(features)
        return disease, features
    elif organ == 'Liver-Tumor':
        disease, features=liver_tumor_model(image_path, filename, "liver_tumor_segmentation_model.pt")
        # print(disease)
        # print(features)
        # print("Here")
        return disease, features
    elif organ == 'Brain' or organ=='brain-dcm':
        # print("starting...")
        disease, features=brain_model(image_path, filename,"Brain_Tumor_Classification_best.pt")
        # print(disease)
        # print(features)
        return disease, features
    elif organ == 'Eye':
        disease=eye_model(image_path)
    elif organ == 'Kidney':
        disease=kidney_model(image_path)
    return disease
   