# 1. Base Image: Use the standard Python image (not slim)
# This includes all the common C++ libraries we need by default.
FROM python:3.9

# 2. Set working directory
WORKDIR /app

# 3. Install Python Libraries
# We removed the 'apt-get' block entirely because this Base Image
# already has the necessary system support.
RUN pip install --no-cache-dir \
    numpy \
    onnxruntime \
    opencv-python-headless

# 4. Copy files
COPY . /app

# 5. Run command
CMD ["python", "inference.py"]