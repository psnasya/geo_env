# GeoLocNet_Plus.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sqlite3
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎯 Используемое устройство: {device}")


class UniversalGeoDataset(Dataset):
    def __init__(self, db_path, transform=None, sample_size=None, data_types=None, indices=None):
        self.db_path = db_path
        self.transform = transform
        self.conn = sqlite3.connect(db_path)

        # Базовый запрос
        query = """
        SELECT file_path, latitude, longitude, camera_id, label, data_source, metadata
        FROM unified_photos 
        WHERE has_image = 1 
        AND latitude IS NOT NULL 
        AND longitude IS NOT NULL
        AND latitude BETWEEN 55.0 AND 56.0
        AND longitude BETWEEN 37.0 AND 38.0
        """

        # Фильтр по типам данных если указан
        if data_types:
            placeholders = ','.join(['?'] * len(data_types))
            query += f" AND data_source IN ({placeholders})"
            params = data_types
        else:
            params = []

        if sample_size:
            query += f" LIMIT {sample_size}"

        self.data = pd.read_sql(query, self.conn, params=params)
        self.conn.close()

        # Если указаны индексы, берем подмножество
        if indices is not None:
            self.data = self.data.iloc[indices].reset_index(drop=True)

        # Анализ данных
        self._analyze_dataset()

        # Нормализация координат
        self.lats = self.data['latitude'].values
        self.lons = self.data['longitude'].values

        self.scaler_lat = StandardScaler()
        self.scaler_lon = StandardScaler()

        self.lats_scaled = self.scaler_lat.fit_transform(self.lats.reshape(-1, 1)).flatten()
        self.lons_scaled = self.scaler_lon.fit_transform(self.lons.reshape(-1, 1)).flatten()

    def _analyze_dataset(self):
        """Анализ состава датасета"""
        print("📊 АНАЛИЗ ДАТАСЕТА:")
        print("=" * 40)

        # По источникам данных
        source_stats = self.data['data_source'].value_counts()
        print("📈 По источникам данных:")
        for source, count in source_stats.items():
            print(f"   {source}: {count} записей")

        # По меткам
        if 'label' in self.data.columns:
            label_stats = self.data['label'].value_counts()
            print("🏷️ По меткам:")
            for label, count in label_stats.items():
                print(f"   {label}: {count} записей")

        # Географическое распределение
        print("📍 Географическое распределение:")
        print(f"   Широта: {self.data['latitude'].min():.3f} - {self.data['latitude'].max():.3f}")
        print(f"   Долгота: {self.data['longitude'].min():.3f} - {self.data['longitude'].max():.3f}")

        print(f"📊 Всего записей в датасете: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        try:
            # Пробуем несколько путей для загрузки изображения
            possible_paths = [
                row['file_path'],  # Основной путь
                row['file_path'].replace('images/', 'extracted_images/'),  # extracted_images
                f"extracted_images/{json.loads(row['metadata'])['data_type']}/{json.loads(row['metadata'])['month']}/{os.path.basename(row['file_path'])}"
            ]

            image = None
            for img_path in possible_paths:
                if os.path.exists(img_path):
                    try:
                        image = Image.open(img_path).convert('RGB')
                        break
                    except:
                        continue

            # Если изображение не найдено, создаем заглушку
            if image is None:
                try:
                    meta_dict = json.loads(row['metadata']) if row['metadata'] else {}
                    data_type = meta_dict.get('data_type', 'unknown')

                    if 'violations' in str(data_type):
                        color = (220, 100, 100)  # Красный
                    elif 'construction' in str(data_type):
                        color = (100, 100, 220)  # Синий
                    elif 'garbage' in str(data_type):
                        color = (100, 200, 100)  # Зеленый
                    else:
                        color = (150, 150, 150)  # Серый

                    image = Image.new('RGB', (224, 224), color=color)
                except:
                    image = Image.new('RGB', (224, 224), color=(128, 128, 128))

            # Преобразуем изображение
            image = np.array(image)
            image = cv2.resize(image, (224, 224))
            image = image.astype(np.float32) / 255.0

            if self.transform:
                image = self.transform(image)

            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()

            # Нормализованные координаты
            coords = torch.tensor([self.lats_scaled[idx], self.lons_scaled[idx]], dtype=torch.float32)

            return image_tensor, coords

        except Exception as e:
            print(f"⚠️ Ошибка загрузки {row['file_path']}: {e}")
            # Fallback
            return torch.rand(3, 224, 224), torch.tensor([0.0, 0.0])


class GeoLocNet_Plus(nn.Module):
    def __init__(self, use_attention=True, use_coord_attention=True):
        super().__init__()
        self.use_attention = use_attention
        self.use_coord_attention = use_coord_attention

        # 🔥 УЛУЧШЕННАЯ АРХИТЕКТУРА
        # Backbone - более глубокая сеть
        self.backbone = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.1),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )

        # 🔥 COORDINATE ATTENTION (новый механизм)
        if use_coord_attention:
            self.coord_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(512, 32, 1),
                nn.ReLU(),
                nn.Conv2d(32, 512, 1),
                nn.Sigmoid()
            )

        # 🔥 SPATIAL ATTENTION
        if use_attention:
            self.spatial_attention = nn.Sequential(
                nn.Conv2d(512, 64, 1), nn.ReLU(),
                nn.Conv2d(64, 1, 1), nn.Sigmoid()
            )

        # 🔥 УЛУЧШЕННЫЙ РЕГРЕССОР
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024), nn.ReLU(), nn.BatchNorm1d(1024), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.BatchNorm1d(512), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)  # latitude, longitude
        )

        # Инициализация весов
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Backbone features
        features = self.backbone(x)

        # Coordinate Attention
        if self.use_coord_attention:
            coord_att = self.coord_attention(features)
            features = features * coord_att

        # Spatial Attention
        if self.use_attention:
            spatial_att = self.spatial_attention(features)
            features = features * spatial_att

        # Regression
        output = self.regressor(features)
        return output


class GeoLocTrainer:
    def __init__(self, db_path='correct_unified_database.db'):
        self.db_path = db_path
        self.model = None
        self.scaler_lat = None
        self.scaler_lon = None
        self.training_history = {
            'train_losses': [], 'val_losses': [], 'learning_rates': [],
            'train_errors_km': [], 'val_errors_km': [], 'timestamps': []
        }

    def prepare_data(self, train_size=45000, val_size=7000, test_size=7000, data_types=None):
        """Подготовка данных с улучшенной аугментацией"""
        print("📊 ПОДГОТОВКА ДАННЫХ ДЛЯ GeoLocNet-Plus")
        print("=" * 50)

        full_dataset = UniversalGeoDataset(self.db_path, data_types=data_types)

        if len(full_dataset) == 0:
            raise ValueError("❌ Нет данных для обучения")

        self.scaler_lat = full_dataset.scaler_lat
        self.scaler_lon = full_dataset.scaler_lon

        # 🔥 БОЛЬШЕ ДАННЫХ ДЛЯ ТРЕНИРОВКИ
        max_samples = min(train_size + val_size + test_size, len(full_dataset))
        indices = np.random.choice(len(full_dataset), max_samples, replace=False)

        train_indices, temp_indices = train_test_split(indices, test_size=val_size + test_size, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=test_size, random_state=42)

        train_dataset = UniversalGeoDataset(self.db_path, data_types=data_types, indices=train_indices)
        val_dataset = UniversalGeoDataset(self.db_path, data_types=data_types, indices=val_indices)
        test_dataset = UniversalGeoDataset(self.db_path, data_types=data_types, indices=test_indices)

        print(f"🎯 Размеры: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset

    def calculate_distance_km(self, lat1, lon1, lat2, lon2):
        """Точный расчет расстояния"""
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(
            dlon / 2) * np.sin(dlon / 2)
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def train(self, epochs=25, batch_size =64, learning_rate=0.0005, data_types=None):
        """Улучшенное обучение с дополнительными метриками"""
        print("🚀 ЗАПУСК ОБУЧЕНИЯ GeoLocNet-Plus")
        print("=" * 60)

        start_time = time.time()

        # Подготовка данных
        train_dataset, val_dataset, test_dataset = self.prepare_data(data_types=data_types)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        # 🔥 НОВАЯ МОДЕЛЬ
        self.model = GeoLocNet_Plus(use_attention=True, use_coord_attention=True).to(device)

        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🧠 GeoLocNet-Plus: {total_params:,} параметров")

        # 🔥 УЛУЧШЕННЫЙ ОПТИМИЗАТОР
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.001)

        # 🔥 COSINE ANNEALING SCHEDULER
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

        # 🔥 КОМБИНИРОВАННАЯ ФУНКЦИЯ ПОТЕРЬ
        criterion1 = nn.HuberLoss()  # Устойчивая к выбросам
        criterion2 = nn.MSELoss()  # Точная регрессия

        best_val_loss = float('inf')
        best_val_error = float('inf')
        best_epoch = 0
        patience = 8

        print(f"🎯 Обучение на {len(train_dataset)} изображениях")
        print(f"📊 Batch size: {batch_size}, Learning rate: {learning_rate}")

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # Training
            self.model.train()
            train_loss = 0
            train_errors = []

            train_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]')
            for images, coords in train_bar:
                images, coords = images.to(device), coords.to(device)

                optimizer.zero_grad()
                outputs = self.model(images)

                # 🔥 КОМБИНИРОВАННАЯ ПОТЕРЯ
                loss1 = criterion1(outputs, coords)
                loss2 = criterion2(outputs, coords)
                loss = 0.7 * loss1 + 0.3 * loss2  # Взвешенная сумма

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()

                # Расчет ошибки в км для батча
                with torch.no_grad():
                    for i in range(outputs.shape[0]):
                        pred = outputs[i].cpu().numpy()
                        true = coords[i].cpu().numpy()
                        pred_lat = self.scaler_lat.inverse_transform([pred[0].reshape(-1)])[0][0]
                        pred_lon = self.scaler_lon.inverse_transform([pred[1].reshape(-1)])[0][0]
                        true_lat = self.scaler_lat.inverse_transform([true[0].reshape(-1)])[0][0]
                        true_lon = self.scaler_lon.inverse_transform([true[1].reshape(-1)])[0][0]
                        error_km = self.calculate_distance_km(true_lat, true_lon, pred_lat, pred_lon)
                        train_errors.append(error_km)

                train_bar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                  #  'AvgError': f'{np.mean(train_errors[-len(images):]):.2f}km',
                    'LR': f'{scheduler.get_last_lr()[0]:.6f}'
                })

            # Validation
            self.model.eval()
            val_loss = 0
            val_errors = []

            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]')
                for images, coords in val_bar:
                    images, coords = images.to(device), coords.to(device)
                    outputs = self.model(images)

                    loss1 = criterion1(outputs, coords)
                    loss2 = criterion2(outputs, coords)
                    loss = 0.7 * loss1 + 0.3 * loss2
                    val_loss += loss.item()

                    # Расчет ошибки для валидации
                    for i in range(outputs.shape[0]):
                        pred = outputs[i].cpu().numpy()
                        true = coords[i].cpu().numpy()
                        pred_lat = self.scaler_lat.inverse_transform([pred[0].reshape(-1)])[0][0]
                        pred_lon = self.scaler_lon.inverse_transform([pred[1].reshape(-1)])[0][0]
                        true_lat = self.scaler_lat.inverse_transform([true[0].reshape(-1)])[0][0]
                        true_lon = self.scaler_lon.inverse_transform([true[1].reshape(-1)])[0][0]
                        error_km = self.calculate_distance_km(true_lat, true_lon, pred_lat, pred_lon)
                        val_errors.append(error_km)

                    val_bar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'AvgError': f'{np.mean(val_errors[-len(images):]):.2f}km'
                    })

            scheduler.step()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_error_km = np.mean(train_errors)
            val_error_km = np.mean(val_errors)

            # Сохраняем историю
            self.training_history['train_losses'].append(train_loss)
            self.training_history['val_losses'].append(val_loss)
            self.training_history['train_errors_km'].append(train_error_km)
            self.training_history['val_errors_km'].append(val_error_km)
            self.training_history['learning_rates'].append(scheduler.get_last_lr()[0])
            self.training_history['timestamps'].append(datetime.now().strftime("%H:%M:%S"))

            epoch_time = time.time() - epoch_start_time
            current_lr = scheduler.get_last_lr()[0]

            print(f'\nEpoch {epoch + 1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            print(f'  Train Error: {train_error_km:.2f}km, Val Error: {val_error_km:.2f}km')
            print(f'  LR: {current_lr:.6f}, Время: {epoch_time:.1f}с')

            # 🔥 УЛУЧШЕННОЕ СОХРАНЕНИЕ (по ошибке в км)
            if val_error_km < best_val_error:
                best_val_error = val_error_km
                best_val_loss = val_loss
                best_epoch = epoch + 1

                self._save_model('GeoLocNet_Plus.pth', best_epoch, best_val_error, best_val_loss)
                print(f'  💾 Сохранена ЛУЧШАЯ модель (epoch {best_epoch}, error: {best_val_error:.2f}km)')

                # Ранний успех
                if best_val_error <= 1.0:
                    print(f"  🎯 ДОСТИГНУТА ЦЕЛЬ 1 КМ!")
                    break
            else:
                epochs_without_improvement = epoch + 1 - best_epoch
                print(f'  📊 Лучшая модель: epoch {best_epoch}, error: {best_val_error:.2f}km')
                print(f'  ⏳ Эпох без улучшений: {epochs_without_improvement}/{patience}')

                if epochs_without_improvement >= patience:
                    print(f"\n🛑 РАННЯЯ ОСТАНОВКА: нет улучшений {patience} эпох")
                    break

        total_time = time.time() - start_time

        print(f"\n🏆 ИТОГИ ОБУЧЕНИЯ GeoLocNet-Plus:")
        print("=" * 50)
        print(f"   Лучшая эпоха: {best_epoch}")
        print(f"   Лучшая ошибка: {best_val_error:.2f} км")
        print(f"   Лучший Val Loss: {best_val_loss:.4f}")
        print(f"   Общее время: {total_time / 60:.1f} минут")

        # Финальное тестирование
        print("\n🔍 ФИНАЛЬНОЕ ТЕСТИРОВАНИЕ")
        final_error = self.evaluate_model(test_loader)

        # Сохраняем финальную модель
        if final_error <= best_val_error:
            self._save_model('GeoLocNet_Plus_Final.pth', best_epoch, final_error, best_val_loss)
            print(f"💾 Сохранена финальная модель с ошибкой: {final_error:.2f}km")

        self.plot_training()

    def _save_model(self, filename, best_epoch, best_error, best_loss):
        """Сохранение модели с улучшенными метаданными"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'scaler_lat_mean': self.scaler_lat.mean_,
            'scaler_lat_scale': self.scaler_lat.scale_,
            'scaler_lon_mean': self.scaler_lon.mean_,
            'scaler_lon_scale': self.scaler_lon.scale_,
            'model_architecture': 'GeoLocNet_Plus',
            'training_history': self.training_history,
            'best_epoch': best_epoch,
            'best_error_km': best_error,
            'best_val_loss': best_loss,
            'save_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'device_used': str(device)
        }

        torch.save(checkpoint, filename)

    def evaluate_model(self, test_loader):
        """Детальная оценка модели"""
        self.model.eval()
        distances_km = []

        with torch.no_grad():
            test_bar = tqdm(test_loader, desc="Testing GeoLocNet-Plus")
            for images, coords in test_bar:
                images, coords = images.to(device), coords.to(device)
                outputs = self.model(images)

                for i in range(outputs.shape[0]):
                    pred = outputs[i].cpu().numpy()
                    true = coords[i].cpu().numpy()
                    pred_lat = self.scaler_lat.inverse_transform([pred[0].reshape(-1)])[0][0]
                    pred_lon = self.scaler_lon.inverse_transform([pred[1].reshape(-1)])[0][0]
                    true_lat = self.scaler_lat.inverse_transform([true[0].reshape(-1)])[0][0]
                    true_lon = self.scaler_lon.inverse_transform([true[1].reshape(-1)])[0][0]

                    error_km = self.calculate_distance_km(true_lat, true_lon, pred_lat, pred_lon)
                    distances_km.append(error_km)

                current_avg = np.mean(distances_km[-len(images):])
                test_bar.set_postfix({'Avg Error': f'{current_avg:.2f} km'})

        distances_km = np.array(distances_km)

        print(f"\n📊 РЕЗУЛЬТАТЫ GeoLocNet-Plus:")
        print("=" * 40)
        print(f"   Средняя ошибка: {np.mean(distances_km):.2f} км")
        print(f"   Медианная ошибка: {np.median(distances_km):.2f} км")
        print(f"   Процентили:")
        for p in [25, 50, 75, 90, 95]:
            error = np.percentile(distances_km, p)
            print(f"     {p}%: {error:.2f} км")

        return np.mean(distances_km)

    def plot_training(self):
        """Визуализация прогресса"""
        plt.figure(figsize=(15, 5))

        epochs_range = range(1, len(self.training_history['train_losses']) + 1)

        # График потерь
        plt.subplot(1, 3, 1)
        plt.plot(epochs_range, self.training_history['train_losses'], label='Train Loss', linewidth=2)
        plt.plot(epochs_range, self.training_history['val_losses'], label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)

        # График ошибок (км)
        plt.subplot(1, 3, 2)
        plt.plot(epochs_range, self.training_history['train_errors_km'], label='Train Error', linewidth=2)
        plt.plot(epochs_range, self.training_history['val_errors_km'], label='Val Error', linewidth=2)
        plt.axhline(y=1.0, color='red', linestyle='--', label='Target 1km')
        plt.xlabel('Epoch')
        plt.ylabel('Error (km)')
        plt.title('Prediction Error')
        plt.legend()
        plt.grid(True)

        # График learning rate
        plt.subplot(1, 3, 3)
        plt.plot(epochs_range, self.training_history['learning_rates'], label='LR', linewidth=2, color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')

        plt.tight_layout()
        plt.savefig('GeoLocNet_Plus_training.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📈 График обучения сохранен: GeoLocNet_Plus_training.png")


def main():
    print("🎯 ЗАПУСК GEOLOCNET-PLUS - ТОЧНАЯ ГЕОЛОКАЦИЯ")
    print("=" * 70)
    print(f"💻 Устройство: {device}")

    # Проверяем наличие данных
    if not os.path.exists('correct_unified_database.db'):
        print("❌ База данных не найдена! Сначала запустите процесс импорта.")
        return

    # Запускаем обучение новой модели
    trainer = GeoLocTrainer()

    try:
        trainer.train(
            epochs=25,  # Больше эпох для лучшей сходимости
            batch_size=32,  # Больше batch size для стабильности
            learning_rate=0.0005,  # Меньше learning rate для точности
            data_types=['excel_import']
        )

        print("\n" + "🎯" * 20)
        print("GEOlocNet-Plus ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print("ЦЕЛЬ: ошибка < 1 км")
        print("🎯" * 20)

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()