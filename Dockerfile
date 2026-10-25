FROM python:3.10-slim

WORKDIR /app

# Install system dependencies including ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy repository code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck endpoint
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit Web Studio
CMD ["streamlit", "run", "apps/web_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
