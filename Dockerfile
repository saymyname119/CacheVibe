FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the setup script to prepare data, then start the server.
# In production, the data directory would be a mounted volume.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
