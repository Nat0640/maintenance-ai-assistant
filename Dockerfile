# ใช้ base image ของ Python
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

# ติดตั้ง Dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดและ VectorDB ทั้งหมดเข้าไป
COPY . .

EXPOSE 8000

# คำสั่งรัน Production server
CMD ["gunicorn", "-w", "2", "-k", "uvicorn.workers.UvicornWorker", "main:app", "--bind", "0.0.0.0:8000"]