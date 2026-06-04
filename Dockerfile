FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY model.py dataset.py inference.py app.py sort_model.pt ./

# HuggingFace Spaces requires port 7860
EXPOSE 7860

CMD ["python", "app.py"]
