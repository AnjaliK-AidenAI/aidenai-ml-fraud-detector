FROM python:3.11-slim AS base
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --upgrade pip \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org && \
    pip install -r requirements.txt \
        --trusted-host pypi.org \
        --trusted-host files.pythonhosted.org

COPY app ./app

RUN mkdir -p models

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","8000","--workers","2"]