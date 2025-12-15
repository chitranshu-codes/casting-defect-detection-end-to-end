import cv2
import numpy as np
import onnxruntime as ort

# 1. SETUP: Load the ONNX model
# We use CPUExecutionProvider for now. If we have NVIDIA, use 'CUDAExecutionProvider'
session = ort.InferenceSession("cdm-yolo.onnx", providers=['CPUExecutionProvider'])

# Get the input and output details so we know what shape the model expects
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = session.get_inputs()[0].shape 
# usually [1, 3, 640, 640] -> (Batch_Size, Channels, Height, Width)

print(f"Model loaded! Expecting input shape: {input_shape}")

# 2. PRE-PROCESSING
img = cv2.imread("test-img.jpeg")  # Load image
if img is None:
    print("Error: Image not found.")
    exit()

# A. Resize to model input size (YOLO standard)
img_resized = cv2.resize(img, (512, 512))

# B. Convert Color: OpenCV uses BGR, but models are trained on RGB
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

# C. Normalization: Convert pixel values from 0-255 to 0.0-1.0
img_normalized = img_rgb.astype(np.float32) / 255.0

# D. Transpose: Change "Height, Width, Channels" (640,640,3) 
img_transposed = img_normalized.transpose(2, 0, 1)

# E. Batch Dimension: Add a dimension at the start
input_tensor = np.expand_dims(img_transposed, axis=0)

# 3. INFERENCE (Running the Engine)
print("Running inference...")
outputs = session.run([output_name], {input_name: input_tensor})

# 4. RAW OUTPUT
raw_result = outputs[0]

print("Inference Complete.")
print(f"Output Shape: {raw_result.shape}") # type: ignore
print(f"First few raw values: {raw_result[0][0][:5]}") # type: ignore


# 1. Transpose Output
predictions = np.squeeze(raw_result).T # type: ignore

# Filter variables
conf_threshold = 0.5   # 50% confidence required
iou_threshold = 0.45   # Drop box if it overlaps > 45% with a better box

# Lists to hold candidate detections
boxes = []
confidences = []
class_ids = []

# Get the scaling factor to map back to Original Image size
# img.shape is (Height, Width, Channels)
h_orig, w_orig, _ = img.shape 
x_scale = w_orig / 512
y_scale = h_orig / 512

print("Filtering detections...")

# 2. Loop through the 8400 potential boxes
for i in range(len(predictions)):
    row = predictions[i]
    
    scores = row[4:]
    
    # Find the class with the maximum score
    max_score = np.amax(scores)
    
    # Check if confidence is high enough
    if max_score >= conf_threshold:
        # Get the index of the class (0 = Person, 1 = Bicycle, etc.)
        class_id = np.argmax(scores)
        
        # Extract Box Coordinates (Relative to 640x640 input)
        x_center, y_center, width, height = row[0], row[1], row[2], row[3]
        
        # Convert to Top-Left Corner (x, y) needed for OpenCV
        x_left = int((x_center - width / 2) * x_scale)
        y_left = int((y_center - height / 2) * y_scale)
        
        # Scale Width/Height to original image
        w = int(width * x_scale)
        h = int(height * y_scale)
        
        # Save to lists
        boxes.append([x_left, y_left, w, h])
        confidences.append(float(max_score))
        class_ids.append(class_id)

# 3. Apply Non-Maximum Suppression (NMS)
indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, iou_threshold)

print(f"Detections found after NMS: {len(indices)}")

# DEFINE YOUR CLASSES HERE
class_names = ["blowhole"]

if len(indices) > 0:
    for i in indices.flatten(): # type: ignore
        # Get the box details
        x, y, w, h = boxes[i]
        # Use the integer ID to grab the string name from your list
        class_id_number = class_ids[i]

        # Safety check: make sure the ID is inside our list
        if class_id_number < len(class_names):
         label_name = class_names[class_id_number]
        else:
         label_name = f"Unknown({class_id_number})"

        score = confidences[i]

        # Draw Rectangle
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        
        # Update the text to show the Name instead of the Number
        text = f"{label_name}: {score:.2f}"
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Save or Show result
cv2.imwrite("final_output.jpg", img)
print("Saved result to final_output.jpg")