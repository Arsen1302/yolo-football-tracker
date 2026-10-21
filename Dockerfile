FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV VIRTUAL_ENV=/opt/venv
RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN if [ "$(dpkg --print-architecture)" = "arm64" ]; then \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://pypi.org/simple \
            "torch==2.0.1" "torchvision==0.15.2"; \
    else \
        pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
            --extra-index-url https://pypi.org/simple \
            "torch==2.9.1+cpu" "torchvision==0.24.1+cpu"; \
    fi

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 10 -r requirements.txt
RUN pip install --no-cache-dir --force-reinstall --no-deps \
    "numpy==2.2.6" "pandas==2.2.3"
RUN python -c "import streamlit, numpy, pandas, sys; print(f'python={sys.executable}, numpy={numpy.__version__}, pandas={pandas.__version__}, streamlit={streamlit.__version__}')"

COPY . .

EXPOSE 7860

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-7860} --server.headless=true"]