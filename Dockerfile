FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir ".[app]"
COPY configs ./configs
COPY app ./app
ENV RETINA_CHECKPOINT=/models/best.pt
EXPOSE 8000 8501
CMD ["uvicorn", "retina_lens.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
