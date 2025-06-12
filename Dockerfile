FROM python:3.10-slim

WORKDIR /app

RUN pip intstall --upgrade pip \
    && pip install ultralytics opencv-python
    
COPY . /app
CMD ["python", "detect_image.py"]


