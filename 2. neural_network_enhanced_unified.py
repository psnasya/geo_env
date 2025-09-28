# neural_network_final_training.py это ласт версия модели
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🎯 Используемое устройство: {device}")


class CorrectGeoDataset(Dataset):
    def __init__(self, db_path, transform=None, sample_size=None):
        self.db_path = db_path
        self.transform = transform
        self.conn = sqlite3.connect(db_path)

        # Берем ТОЛЬКО данные с координатами (пока только camera)
        query = """
        SELECT file_path, latitude, longitude 
        FROM unified_photos 
        WHERE has_image = 1 
        AND latitude IS NOT NULL 
        AND longitude IS NOT NULL
        AND latitude BETWEEN 55.0 AND 56.0
        AND longitude BETWEEN 37.0 AND 38.0
        """

        if sample_size:
            query += f" LIMIT {sample_size}"

        self.data = pd.read_sql(query, self.conn)
        self.conn.close()

        # Нормализация координат под Москву
        self.lats = self.data['latitude'].values
        self.lons = self.data['longitude'].values

        # Нормализуем к [0, 1] диапазону
        self.scaler_lat = StandardScaler()
        self.scaler_lon = StandardScaler()

        self.lats_scaled = self.scaler_lat.fit_transform(self.lats.reshape(-1, 1)).flatten()
        self.lons_scaled = self.scaler_lon.fit_transform(self.lons.reshape(-1, 1)).flatten()

        print(f"📊 Загружено {len(self.data)} изображений с координатами")
        print(
            f"📍 Диапазон координат: {self.lats.min():.3f}-{self.lats.max():.3f}, {self.lons.min():.3f}-{self.lons.max():.3f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        try:
            # Загрузка и предобработка изображения
            image = Image.open(row['file_path']).convert('RGB')
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
            # Fallback на случай ошибки
            return torch.rand(3, 224, 224), torch.tensor([0.0, 0.0])


class EfficientGeoCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2), nn.Dropout(0.3),

            # Block 3
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

        # Инициализация весов
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


class FinalGeoLocationTrainer:
    def __init__(self, db_path='correct_unified_database.db'):
        self.db_path = db_path
        self.model = None
        self.scaler_lat = None
        self.scaler_lon = None

    def prepare_data(self, test_size=0.2, train_size=8000, val_size=2000):
        """Подготовка данных с контролируемым размером"""
        print("📊 ПОДГОТОВКА ДАННЫХ")
        print("=" * 40)

        # Создаем датасет
        full_dataset = CorrectGeoDataset(self.db_path)

        if len(full_dataset) == 0:
            raise ValueError("❌ Нет данных для обучения")

        # Сохраняем scalers для предсказаний
        self.scaler_lat = full_dataset.scaler_lat
        self.scaler_lon = full_dataset.scaler_lon

        # Ограничиваем размер данных для стабильности
        max_samples = min(train_size + val_size, len(full_dataset))
        indices = np.random.choice(len(full_dataset), max_samples, replace=False)

        # Разделение
        train_indices, temp_indices = train_test_split(indices, test_size=val_size, random_state=42)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.5, random_state=42)

        # Создаем сабсеты
        class SubsetDataset(Dataset):
            def __init__(self, dataset, indices):
                self.dataset = dataset
                self.indices = indices

            def __len__(self):
                return len(self.indices)

            def __getitem__(self, idx):
                return self.dataset[self.indices[idx]]

        train_dataset = SubsetDataset(full_dataset, train_indices)
        val_dataset = SubsetDataset(full_dataset, val_indices)
        test_dataset = SubsetDataset(full_dataset, test_indices)

        print(f"🎯 Размеры: Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

        return train_dataset, val_dataset, test_dataset

    def train(self, epochs=10, batch_size=16, learning_rate=0.001):
        """Обучение модели"""
        print("🚀 ЗАПУСК ОБУЧЕНИЯ")
        print("=" * 40)

        # Подготовка данных
        train_dataset, val_dataset, test_dataset = self.prepare_data()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Модель и оптимизатор
        self.model = EfficientGeoCNN().to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        print(f"🎯 Обучение на {len(train_dataset)} изображениях")

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for images, coords in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
                images, coords = images.to(device), coords.to(device)

                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, coords)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for images, coords in val_loader:
                    images, coords = images.to(device), coords.to(device)
                    val_loss += criterion(self.model(images), coords).item()

            train_loss /= len(train_loader)
            val_loss /= len(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            print(f'Epoch {epoch + 1}/{epochs}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Сохраняем лучшую модель
            if epoch == 0 or val_loss < min(val_losses[:-1]):
                self.save_model('final_geo_model.pth')
                print('💾 Сохранена лучшая модель')

        # Финальная оценка
        self.evaluate_model(test_loader)

        # График обучения
        self.plot_training(train_losses, val_losses)

    def evaluate_model(self, test_loader):
        """Оценка модели"""
        print("\n🔍 ТЕСТИРОВАНИЕ МОДЕЛИ")
        print("=" * 40)

        self.model.eval()
        distances = []

        with torch.no_grad():
            for images, coords in test_loader:
                images, coords = images.to(device), coords.to(device)
                outputs = self.model(images)

                # Денормализация и расчет расстояния
                for i in range(outputs.shape[0]):
                    pred = outputs[i].cpu().numpy()
                    true = coords[i].cpu().numpy()

                    pred_lat = self.scaler_lat.inverse_transform([pred[0].reshape(-1)])[0][0]
                    pred_lon = self.scaler_lon.inverse_transform([pred[1].reshape(-1)])[0][0]
                    true_lat = self.scaler_lat.inverse_transform([true[0].reshape(-1)])[0][0]
                    true_lon = self.scaler_lon.inverse_transform([true[1].reshape(-1)])[0][0]

                    # Расчет расстояния в км
                    distance = self.calculate_distance_km(true_lat, true_lon, pred_lat, pred_lon)
                    distances.append(distance)

        distances = np.array(distances)
        print(f"📊 Результаты на тестовых данных:")
        print(f"   Средняя ошибка: {np.mean(distances):.2f} км")
        print(f"   Медианная ошибка: {np.median(distances):.2f} км")
        print(f"   Лучшая ошибка: {np.min(distances):.2f} км")
        print(f"   Худшая ошибка: {np.max(distances):.2f} км")

    def calculate_distance_km(self, lat1, lon1, lat2, lon2):
        """Расчет расстояния в км"""
        R = 6371
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    def plot_training(self, train_losses, val_losses):
        """График обучения"""
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_progress.png')
        plt.close()
        print("📈 График обучения сохранен: training_progress.png")

    def save_model(self, filename):
        """Сохранение модели"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'scaler_lat_mean': self.scaler_lat.mean_,
            'scaler_lat_scale': self.scaler_lat.scale_,
            'scaler_lon_mean': self.scaler_lon.mean_,
            'scaler_lon_scale': self.scaler_lon.scale_
        }, filename)


def main():
    print("🎯 ФИНАЛЬНОЕ ОБУЧЕНИЕ НЕЙРОСЕТИ")
    print("=" * 50)

    trainer = FinalGeoLocationTrainer()

    try:
        trainer.train(epochs=10, batch_size=16, learning_rate=0.001)

        print("\n" + "🎉" * 10)
        print("ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print("Модель готова к использованию!")
        print("🎉" * 10)

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()