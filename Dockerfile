FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/

RUN python3 -c "import sys; lines=[l.split('--hash')[0].strip() for l in open('requirements.txt') if l.strip()]; open('clean_requirements.txt','w').write('\n'.join(lines))"

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r clean_requirements.txt

COPY . /app

EXPOSE 8501

ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0

CMD ["streamlit", "run", "app.py"]