FROM python:3.9-slim

# Install system dependencies for OpenCV (libgl1 replaces deprecated libgl1-mesa-glx on Debian Trixie+)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Expose port and run with increased upload limit
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
