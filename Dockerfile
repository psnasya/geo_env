FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Копируем и устанавливаем Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем ВСЕ файлы приложения (включая модель)
COPY . .

# Проверяем что файлы скопировались
RUN echo "📁 Содержимое папки /app:" && ls -la

# Проверяем что модель есть
RUN if [ -f "unified_geo_model.pth" ]; then \
        echo "✅ Модель unified_geo_model.pth найдена" && \
        echo "📊 Размер модели:" && du -h unified_geo_model.pth; \
    else \
        echo "❌ Модель unified_geo_model.pth не найдена!" && \
        echo "📁 Доступные .pth файлы:" && ls -la *.pth || echo "Нет .pth файлов"; \
    fi

EXPOSE 8080

CMD ["python", "app.py"]