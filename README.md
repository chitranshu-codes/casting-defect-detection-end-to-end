# Casting Defect Detection End-to-End

This project implements an automated computer vision pipeline to detect manufacturing defects (specifically **blowholes**) in metal casting images. It utilizes a pre-trained YOLO model exported to **ONNX** format for high-performance inference and is containerized using **Docker** for consistent deployment.

## 📌 Project Overview

  * **Goal:** Detect and localize "blowhole" defects in industrial casting imagery.
  * **Model:** YOLOv8 (exported to `.onnx`).
  * **Inference Engine:** ONNX Runtime (CPU Execution).
  * **Post-Processing:** Custom NumPy implementation of Non-Maximum Suppression (NMS) and coordinate scaling.
  * **Deployment:** Docker containerization.

## 📂 Project Structure

```text
├── cdm-yolo.onnx        # The pre-trained ONNX model file
├── Dockerfile           # Configuration for building the Docker image
├── inference.py         # Main Python script for preprocessing & inference
├── test-img.jpeg        # Input image for testing
├── final_output.jpg     # Generated output with bounding boxes
└── README.md            # Project documentation
```

## 🛠️ Prerequisites

To run this project, you need either **Python 3.9+** installed locally or **Docker**.

### Required Files

Ensure you have the following files in your directory before running:

1.  `cdm-yolo.onnx`: Your trained YOLO model.
2.  `test-img.jpeg`: An image of a metal casting to test.

## 🚀 Usage Option 1: Running Locally (Python)

If you prefer to run the script directly on your machine:

1.  **Install Dependencies**
    It is recommended to use a virtual environment.

    ```bash
    pip install numpy onnxruntime opencv-python
    ```

    *(Note: Use `opencv-python-headless` if running in a server environment without a GUI).*

2.  **Run the Inference Script**

    ```bash
    python inference.py
    ```

3.  **View Results**
    The script will generate `final_output.jpg` with the detected defects highlighted.

## 🐳 Usage Option 2: Running with Docker

This project includes a `Dockerfile` to simplify dependency management and deployment.

1.  **Build the Docker Image**
    Run the following command in the project root directory:

    ```bash
    docker build -t casting-detector .
    ```

2.  **Run the Container**
    To run the inference and ensure the output file is saved back to your local machine (using a volume mount):

    ```bash
    # Linux/Mac
    docker run -v $(pwd):/app casting-detector

    # Windows (Command Prompt)
    docker run -v %cd%:/app casting-detector
    ```

    *This maps your current folder to the `/app` folder inside the container, allowing the script to read your image and write the `final_output.jpg` back to your disk.*

## ⚙️ Technical Details

### The Pipeline

The `inference.py` script performs the following steps:

1.  **Preprocessing:**
      * Resizes image to **512x512** (Model Input Shape).
      * Converts BGR to **RGB**.
      * Normalizes pixel values to `0.0 - 1.0`.
      * Transposes dimensions to `(Batch, Channels, Height, Width)`.
2.  **Inference:**
      * Uses `onnxruntime` to run the model on the CPU.
3.  **Post-Processing:**
      * Filters predictions based on a confidence threshold of **0.5**.
      * Applies **Non-Maximum Suppression (NMS)** with an IoU threshold of **0.45** to remove duplicate boxes.
      * Rescales bounding boxes to the original image resolution.

### Why ONNX?

Using ONNX (Open Neural Network Exchange) allows this model to be interoperable across different frameworks and optimized for production environments independent of the original training framework (PyTorch/YOLO).

## 📝 Customization

To detect different defects or classes:

1.  Retrain your model and update `cdm-yolo.onnx`.
2.  Update the `class_names` list in `inference.py`:
    ```python
    class_names = ["blowhole"]
    ```
