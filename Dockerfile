FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .


RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libopencv-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt \
    && mkdir -p /logs

VOLUME /logs

# Copia o restante do código para o container
COPY . .

# Define a variável de ambiente para o Python não gerar arquivos .pyc
ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["python", "-m", "application.networks.pipeline", "-e", "4", "-f", "1"]