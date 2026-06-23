FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required by PyMuPDF
RUN apt-get update && apt-get install -y \
    libmupdf-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY data/sample_docs/ ./data/sample_docs/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
