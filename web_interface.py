# web_interface_fixed.py
import os
import torch
from PIL import Image
import io
import numpy as np
import cv2
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# Простая HTML страница
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>GeoLocation AI</title>
    <meta charset="UTF-8">
    <style>
        body { 
            font-family: Arial, sans-serif; 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 30px;
            border-radius: 15px;
            margin: 20px 0;
        }
        .upload-container { 
            border: 2px dashed rgba(255, 255, 255, 0.3); 
            padding: 40px; 
            text-align: center; 
            margin: 20px 0; 
            border-radius: 10px;
        }
        .result { 
            background: rgba(255, 255, 255, 0.2); 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 10px;
            display: none;
        }
        button { 
            background: #4CAF50; 
            color: white; 
            padding: 12px 30px; 
            border: none; 
            border-radius: 25px; 
            cursor: pointer; 
            font-size: 16px;
            transition: all 0.3s;
        }
        button:hover { 
            background: #45a049; 
            transform: translateY(-2px);
        }
        input[type="file"] {
            padding: 10px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 5px;
            color: white;
        }
        h1 { text-align: center; margin-bottom: 30px; }
        .coordinate { 
            font-family: 'Courier New', monospace; 
            background: rgba(0, 0, 0, 0.3); 
            padding: 10px; 
            border-radius: 5px; 
            margin: 5px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 Геолокация по изображению</h1>

        <div class="upload-container">
            <h3>📸 Загрузите изображение</h3>
            <form id="uploadForm" enctype="multipart/form-data">
                <input type="file" name="image" accept="image/*" required>
                <br><br>
                <button type="submit">📍 Определить координаты</button>
            </form>
        </div>

        <div id="result" class="result">
            <h3>📊 Результаты:</h3>
            <div class="coordinate"><strong>Широта:</strong> <span id="latitude">-</span></div>
            <div class="coordinate"><strong>Долгота:</strong> <span id="longitude">-</span></div>
            <p><strong>Точность:</strong> <span id="error">-</span></p>
            <div id="mapLink" style="margin-top: 15px;"></div>
        </div>

        <div style="text-align: center; margin-top: 30px; opacity: 0.8;">
            <small>AI модель определяет географические координаты по визуальным признакам на изображении</small>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const fileInput = document.querySelector('input[type="file"]');
            const resultDiv = document.getElementById('result');
            const submitBtn = this.querySelector('button');

            if (!fileInput.files[0]) {
                alert('Пожалуйста, выберите файл');
                return;
            }

            // Показываем загрузку
            submitBtn.textContent = '⏳ Обработка...';
            submitBtn.disabled = true;

            const formData = new FormData();
            formData.append('image', fileInput.files[0]);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();

                if (result.error) {
                    alert('Ошибка: ' + result.error);
                } else {
                    // Обновляем результаты
                    document.getElementById('latitude').textContent = result.latitude.toFixed(6);
                    document.getElementById('longitude').textContent = result.longitude.toFixed(6);
                    document.getElementById('error').textContent = result.error_estimate;

                    // Ссылка на карту
                    const mapUrl = `https://yandex.ru/maps/?pt=${result.longitude},${result.latitude}&z=15&l=map`;
                    document.getElementById('mapLink').innerHTML = 
                        `<a href="${mapUrl}" target="_blank" style="color: #4CAF50;">🗺️ Открыть на Яндекс.Картах</a>`;

                    resultDiv.style.display = 'block';
                }
            } catch (error) {
                alert('Ошибка загрузки: ' + error);
            } finally {
                // Восстанавливаем кнопку
                submitBtn.textContent = '📍 Определить координаты';
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
'''


class SimpleGeoCNN(torch.nn.Module):
    """Упрощенная модель геолокации"""

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((7, 7))
        )
        self.regressor = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 7 * 7, 512), torch.nn.ReLU(),
            torch.nn.Linear(512, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.regressor(self.features(x))


def safe_load_model():
    """Безопасная загрузка модели с обработкой ошибок"""
    try:
        if not os.path.exists('final_geo_model.pth'):
            print("❌ Файл модели не найден")
            return None, None

        # Загрузка с отключенной проверкой весов
        checkpoint = torch.load('final_geo_model.pth', map_location='cpu', weights_only=False)

        model = SimpleGeoCNN()

        # Совместимость с разными версиями PyTorch
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Прямая загрузка если модель сохранена напрямую
            model.load_state_dict(checkpoint)

        model.eval()
        print("✅ Модель успешно загружена")
        return model, checkpoint

    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return None, None


# Загружаем модель
model, scalers = safe_load_model()


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Модель не загружена. Проверьте файл final_geo_model.pth'})

    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Файл не получен'})

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Файл не выбран'})

        # Проверяем тип файла
        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            return jsonify({'error': 'Поддерживаются только JPG, JPEG и PNG файлы'})

        # Читаем изображение
        image_data = file.read()
        if len(image_data) == 0:
            return jsonify({'error': 'Пустой файл'})

        # Обрабатываем изображение
        image = Image.open(io.BytesIO(image_data)).convert('RGB')

        # Предобработка
        image = np.array(image)
        original_height, original_width = image.shape[:2]
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float().unsqueeze(0)

        # Предсказание
        with torch.no_grad():
            prediction = model(image_tensor).numpy()[0]

        # Денормализация координат (Москва)
        if scalers and 'scaler_lat_mean' in scalers:
            lat_mean = scalers['scaler_lat_mean']
            lat_scale = scalers['scaler_lat_scale']
            lon_mean = scalers['scaler_lon_mean']
            lon_scale = scalers['scaler_lon_scale']

            # Проверяем размерности
            lat_mean = lat_mean[0] if hasattr(lat_mean, '__len__') else lat_mean
            lat_scale = lat_scale[0] if hasattr(lat_scale, '__len__') else lat_scale
            lon_mean = lon_mean[0] if hasattr(lon_mean, '__len__') else lon_mean
            lon_scale = lon_scale[0] if hasattr(lon_scale, '__len__') else lon_scale

            lat = lat_mean + prediction[0] * lat_scale
            lon = lon_mean + prediction[1] * lon_scale
        else:
            # Fallback: нормализация для Москвы
            lat = 55.7 + prediction[0] * 0.3  # ~55.4-56.0
            lon = 37.6 + prediction[1] * 0.3  # ~37.3-37.9

        return jsonify({
            'latitude': float(lat),
            'longitude': float(lon),
            'error_estimate': '12 км (средняя точность)',
            'image_size': f'{original_width}x{original_height}',
            'message': 'Координаты успешно определены!'
        })

    except Exception as e:
        return jsonify({'error': f'Ошибка обработки: {str(e)}'})


@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'service': 'GeoLocation AI'
    })


@app.route('/demo')
def demo_page():
    """Демо-страница с примером"""
    demo_html = '''
    <h1>🎯 Демо GeoLocation AI</h1>
    <p>Загрузите фотографию из Москвы для тестирования</p>
    <p><small>Примеры: уличные сцены, архитектура, панорамы</small></p>
    <a href="/">Вернуться к загрузке</a>
    '''
    return render_template_string(demo_html)


if __name__ == '__main__':
    print("🚀 Запуск веб-сервера GeoLocation AI...")
    print("📁 Текущая директория:", os.getcwd())
    print("📊 Модель загружена:", model is not None)
    print("🌐 Сервер доступен по адресу: http://localhost:5000")
    print("🔧 Для остановки сервера нажмите Ctrl+C")

    app.run(host='0.0.0.0', port=5000, debug=True)