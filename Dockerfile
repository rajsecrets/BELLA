FROM python:3.9-slim

WORKDIR /app

# Copy all files
COPY . /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user and give permissions to /app
RUN useradd -m myuser && chown -R myuser:myuser /app
USER myuser

# Expose port 7860
EXPOSE 7860

# Run the app
CMD ["chainlit", "run", "app.py", "--port", "7860"]