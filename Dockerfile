FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libxcb1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    docling \
    rapidocr-onnxruntime

RUN pip install ultralytics huggingface_hub opencv-python pillow

COPY src/ ./

COPY ["best.pt","huyvux3005/manga109-segmentation-bubble/best.pt"]

COPY ["One Piece/Chapter 1/012.png", "page001.png"]
CMD ["python", "-u", "process.py"]