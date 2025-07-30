# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory if it doesn't exist
RUN mkdir -p data/models

# Expose ports
EXPOSE 5000 8501

# Set environment variables
ENV PYTHONPATH=/app
ENV FLASK_APP=app/drone_detection.py

# Copy startup script
COPY start_services.sh .
RUN chmod +x start_services.sh

# Default command
CMD ["./start_services.sh"]
