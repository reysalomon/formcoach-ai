# Контейнер для выкладки мобильного сайта в интернет (например, Hugging Face Spaces).
FROM python:3.12-slim

# Графические библиотеки, без которых MediaPipe не запускается
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgles2 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face запускает контейнер от пользователя с id 1000
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .

ENV PATH="/home/user/.local/bin:$PATH"
EXPOSE 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
