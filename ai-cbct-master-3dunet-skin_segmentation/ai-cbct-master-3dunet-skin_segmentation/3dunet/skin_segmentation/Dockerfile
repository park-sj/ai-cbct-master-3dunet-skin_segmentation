FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul
ENV LANG=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt update && \
    apt install -y tzdata locales python3-pip python3-venv git libcusolver-dev-11-3 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --upgrade -r requirements.txt
