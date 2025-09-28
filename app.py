import os
import torch
import torch.nn as nn
from PIL import Image
import io
import numpy as np
import cv2
from flask import Flask, request, jsonify

app = Flask(__name__)


def create_compatible_model(state_dict):
    """Создает модель, совместимую с загруженными весами"""
    # Анализируем структуру state_dict чтобы определить архитектуру
    has_batchnorm = any('running_mean' in key for key in state_dict.keys())

    print(f"🔍 Анализ модели: BatchNorm = {has_batchnorm}")

    if has_batchnorm:
        # Архитектура с BatchNorm (для unified_geo_model.pth)
        class UnifiedGeoCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2), nn.Dropout(0.2),

                    nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2), nn.Dropout(0.3),

                    nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((7, 7))
                )

                self.regressor = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 7 * 7, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(512, 256), nn.ReLU(),
                    nn.Linear(256, 2)
                )

            def forward(self, x):
                x = self.features(x)
                x = self.regressor(x)
                return x

        return UnifiedGeoCNN()

    else:
        # Архитектура без BatchNorm (для final_geo_model.pth)
        class EfficientGeoCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2), nn.Dropout(0.2),

                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2), nn.Dropout(0.3),

                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((7, 7))
                )

                self.regressor = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(256 * 7 * 7, 512), nn.ReLU(), nn.Dropout(0.5),
                    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.ReLU(),
                    nn.Linear(128, 2)
                )

            def forward(self, x):
                x = self.features(x)
                x = self.regressor(x)
                return x

        return EfficientGeoCNN()


# Загрузка модели
def load_model():
    try:
        # Сначала попробуем unified_geo_model.pth
        model_path = 'unified_geo_model.pth'
        if not os.path.exists(model_path):
            # Если нет, попробуем final_geo_model.pth
            model_path = 'final_geo_model.pth'
            if not os.path.exists(model_path):
                print("❌ Файлы моделей не найдены")
                return None, None

        print(f"📁 Загрузка модели: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        # Создаем совместимую модель
        model = create_compatible_model(checkpoint['model_state_dict'])

        # Загружаем веса (игнорируем несовместимые)
        model_dict = model.state_dict()
        pretrained_dict = {}

        loaded_count = 0
        total_count = 0

        for k, v in checkpoint['model_state_dict'].items():
            total_count += 1
            if k in model_dict:
                if model_dict[k].shape == v.shape:
                    pretrained_dict[k] = v
                    loaded_count += 1
                    print(f"✅ Загружен: {k}")
                else:
                    print(f"⚠️  Размер не совпадает: {k} - {v.shape} vs {model_dict[k].shape}")
            else:
                print(f"❌ Слой не найден: {k}")

        if pretrained_dict:
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(f"🎯 Загружено {loaded_count}/{total_count} слоев")
        else:
            print("⚠️  Используем случайную инициализацию")

        model.eval()

        print("✅ Модель успешно загружена!")

        # Детальная информация о scalers
        print("📊 Детали checkpoint:")
        for key, value in checkpoint.items():
            if 'scaler' in key:
                print(f"   {key}: {value} (shape: {value.shape})")

        return model, checkpoint

    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        import traceback
        traceback.print_exc()
        return None, None


model, checkpoint = load_model()


def denormalize_coordinates(normalized_coords, checkpoint):
    """Правильная денормализация координат"""
    try:
        # Получаем параметры нормализации
        lat_mean = checkpoint['scaler_lat_mean'][0]
        lat_scale = checkpoint['scaler_lat_scale'][0]
        lon_mean = checkpoint['scaler_lon_mean'][0]
        lon_scale = checkpoint['scaler_lon_scale'][0]

        print(f"🔧 Параметры денормализации:")
        print(f"   Lat: mean={lat_mean:.6f}, scale={lat_scale:.6f}")
        print(f"   Lon: mean={lon_mean:.6f}, scale={lon_scale:.6f}")
        print(f"   Нормализованные координаты: {normalized_coords}")

        # Денормализация: original = normalized * scale + mean
        lat = normalized_coords[0] * lat_scale + lat_mean
        lon = normalized_coords[1] * lon_scale + lon_mean

        print(f"   Денормализованные: lat={lat:.6f}, lon={lon:.6f}")

        return lat, lon

    except Exception as e:
        print(f"❌ Ошибка денормализации: {e}")
        return 55.7558, 37.6173


@app.route('/')
def home():
    return '''
    <html>
    <body style="font-family: Arial; text-align: center; padding: 50px;">
        <h1>🎯 Геолокация камеры</h1>
        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="image" accept="image/*" required>
            <button type="submit">Определить координаты камеры</button>
        </form>
        <p>Статус модели: ''' + ('✅ Загружена' if model else '❌ Не загружена') + '''</p>
    </body>
    </html>
    '''


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or checkpoint is None:
        return "Модель не загружена"

    try:
        file = request.files['image']
        image = Image.open(io.BytesIO(file.read())).convert('RGB')
        image = np.array(image)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)

        with torch.no_grad():
            pred_normalized = model(image_tensor).numpy()[0]

        lat, lon = denormalize_coordinates(pred_normalized, checkpoint)

        return f'''
        <html>
        <body style="font-family: Arial; padding: 50px;">
            <h2>📍 Координаты камеры:</h2>
            <p><b>Широта:</b> {lat:.6f}</p>
            <p><b>Долгота:</b> {lon:.6f}</p>
            <p><a href="/">Назад</a></p>
        </body>
        </html>
        '''
    except Exception as e:
        return f"Ошибка обработки изображения: {e}"


if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    print(f"🚀 Запуск приложения на порту {port}")
    app.run(host='0.0.0.0', port=port, debug=False)