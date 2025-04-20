# Use Python 3.9 base image
FROM python:3.9-slim-bullseye

# Install system dependencies for MediaPipe/OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Start the app with Gunicorn
CMD ["gunicorn", "app:app", "-b", "0.0.0.0:5000"]
