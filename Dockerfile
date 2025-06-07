FROM python:3.9-slim

# Install system dependencies including FFmpeg
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements_hf.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables for Hugging Face Spaces
ENV PORT=7860
ENV HOST=0.0.0.0
ENV FLASK_ENV=production
ENV FLASK_DEBUG=False

# Expose the port
EXPOSE 7860

# Create necessary directories
RUN mkdir -p uploads temp

# Run the application
CMD ["python", "frontend/run.py"] 