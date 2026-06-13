FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY model_engineering/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_engineering/ .

ENV DATA_DIR=/data
ENV PYTHONPATH=/app

ENTRYPOINT ["python", "main.py"]
