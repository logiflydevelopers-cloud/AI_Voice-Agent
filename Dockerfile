FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (ffmpeg needed for audio)
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# 🚀 Start LiveKit agent in production mode
CMD ["python", "main.py", "start"]
