# 2.1_unpack_and_organize_fixed.py анализ структуры
import os
import json
import zipfile
import sqlite3
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import shutil


class WindowsDataOrganizer:
    def __init__(self, base_path):
        self.base_path = base_path
        # Используем короткие пути для избежания ограничений Windows
        self.processed_path = os.path.join(base_path, "organized_data")
        self.db_path = 'camera_geolocation_expanded.db'

        self.create_folder_structure()

    def create_folder_structure(self):
        """Создает упрощенную структуру папок для Windows"""
        # Короткие имена папок
        folders = [
            "cs_18_001/july/img",  # construction_sites
            "cs_18_001/july/meta",
            "cs_18_001/august/img",
            "cs_18_001/august/meta",
            "ncb_00_022/july/img",  # non_compliant_buildings
            "ncb_00_022/july/meta",
            "ncb_00_022/august/img",
            "ncb_00_022/august/meta"
        ]

        for folder in folders:
            os.makedirs(os.path.join(self.processed_path, folder), exist_ok=True)

        print("📁 Создана упрощенная структура папок для Windows")

    def safe_extract_zip(self, zip_file, target_dir):
        """Безопасное извлечение ZIP с обработкой длинных путей"""
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                # Получаем список файлов
                file_list = zip_ref.namelist()
                image_files = [f for f in file_list if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                print(f"   📊 Найдено {len(image_files)} изображений в архиве")

                success_count = 0
                error_count = 0

                for file_path in tqdm(image_files, desc="Извлечение"):
                    try:
                        # Извлекаем только имя файла (без пути)
                        filename = os.path.basename(file_path)

                        # Создаем безопасный путь
                        safe_target_path = os.path.join(target_dir, filename)

                        # Проверяем длину пути
                        if len(safe_target_path) > 240:
                            print(f"   ⚠️ Слишком длинный путь: {filename}")
                            continue

                        # Извлекаем файл
                        with zip_ref.open(file_path) as source_file:
                            with open(safe_target_path, 'wb') as target_file:
                                shutil.copyfileobj(source_file, target_file)

                        success_count += 1

                    except Exception as e:
                        error_count += 1
                        if error_count <= 5:  # Показываем только первые 5 ошибок
                            print(f"   ⚠️ Ошибка извлечения {os.path.basename(file_path)}: {e}")

                print(f"   ✅ Успешно извлечено: {success_count} файлов")
                print(f"   ❌ Ошибок: {error_count} файлов")

                return success_count, error_count

        except Exception as e:
            print(f"❌ Критическая ошибка архива {zip_file}: {e}")
            return 0, 0

    def unpack_and_organize_archives(self):
        """Распаковывает архивы с учетом ограничений Windows"""
        print("📦 РАСПАКОВКА АРХИВОВ (Windows-совместимая)")
        print("=" * 60)

        zip_files = list(Path(self.base_path).rglob('*.zip'))
        total_stats = {'success': 0, 'errors': 0}

        for zip_file in zip_files:
            try:
                # Определяем тип и месяц
                if 'строительная площадка' in zip_file.name.lower():
                    obj_type = 'cs_18_001'  # Короткое имя
                    obj_name = 'construction_sites'
                else:
                    obj_type = 'ncb_00_022'  # Короткое имя
                    obj_name = 'non_compliant_buildings'

                if 'июль' in zip_file.name.lower():
                    month = 'july'
                else:
                    month = 'august'

                # Путь для распаковки
                target_dir = os.path.join(self.processed_path, obj_type, month, 'img')

                print(f"\n📦 Обрабатываем: {zip_file.name}")
                print(f"   🏷️  Тип: {obj_name}")
                print(f"   📅 Месяц: {month}")
                print(f"   📂 Целевая папка: {target_dir}")

                # Распаковываем
                success, errors = self.safe_extract_zip(zip_file, target_dir)
                total_stats['success'] += success
                total_stats['errors'] += errors

            except Exception as e:
                print(f"❌ Ошибка обработки архива {zip_file.name}: {e}")

        print(f"\n📊 ИТОГИ РАСПАКОВКИ:")
        print(f"   ✅ Успешно извлечено: {total_stats['success']} файлов")
        print(f"   ❌ Ошибок: {total_stats['errors']} файлов")

    def process_json_metadata_simple(self):
        """Упрощенная обработка метаданных для Windows"""
        print("\n🔗 ОБРАБОТКА МЕТАДАННЫХ")
        print("=" * 50)

        json_files = list(Path(self.base_path).rglob('*.json'))
        all_metadata = []

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Определяем тип и месяц
                if 'строительная площадка' in json_file.name.lower():
                    obj_type = 'cs_18_001'
                    obj_name = 'construction_sites'
                else:
                    obj_type = 'ncb_00_022'
                    obj_name = 'non_compliant_buildings'

                if 'июль' in json_file.name.lower():
                    month = 'july'
                else:
                    month = 'august'

                print(f"📄 Обрабатываем: {json_file.name}")

                if 'results' in data and isinstance(data['results'], list):
                    # Обрабатываем только первые 1000 записей для теста
                    sample_size = min(1000, len(data['results']))

                    for i, item in enumerate(tqdm(data['results'][:sample_size], desc="Записи")):
                        photo_id = item.get('id', 'unknown')

                        # Создаем упрощенную запись
                        metadata_record = {
                            'photo_id': photo_id,
                            'object_type': obj_type,
                            'month': month,
                            'filename': f"{photo_id}.jpg",
                            'expected_path': os.path.join(self.processed_path, obj_type, month, 'img',
                                                          f"{photo_id}.jpg"),
                            'has_image': os.path.exists(
                                os.path.join(self.processed_path, obj_type, month, 'img', f"{photo_id}.jpg")),
                            'source_json': Path(json_file).name
                        }

                        all_metadata.append(metadata_record)

                print(f"   ✅ Обработано {sample_size} записей")

            except Exception as e:
                print(f"❌ Ошибка обработки {json_file.name}: {e}")

        # Сохраняем метаданные
        metadata_df = pd.DataFrame(all_metadata)
        metadata_path = os.path.join(self.processed_path, 'metadata.csv')
        metadata_df.to_csv(metadata_path, index=False, encoding='utf-8')

        print(f"💾 Метаданные сохранены: {metadata_path}")

        # Статистика
        total_records = len(metadata_df)
        with_images = metadata_df['has_image'].sum()
        print(f"📊 Статистика: {with_images}/{total_records} записей имеют изображения")

        return metadata_df


class SimpleDatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)

    def create_simple_schema(self):
        """Создает упрощенную схему базы данных"""
        print("\n🗃️  СОЗДАНИЕ УПРОЩЕННОЙ СХЕМЫ БАЗЫ")
        print("=" * 50)

        simple_schema = [
            '''CREATE TABLE IF NOT EXISTS photos_enhanced (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_id TEXT,
                object_type TEXT,
                month TEXT,
                file_path TEXT,
                has_image BOOLEAN,
                latitude REAL,
                longitude REAL,
                processed BOOLEAN DEFAULT FALSE,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''',

            '''CREATE INDEX IF NOT EXISTS idx_photo_id ON photos_enhanced(photo_id)''',
            '''CREATE INDEX IF NOT EXISTS idx_object_type ON photos_enhanced(object_type)''',
            '''CREATE INDEX IF NOT EXISTS idx_has_image ON photos_enhanced(has_image)'''
        ]

        for sql in simple_schema:
            try:
                self.conn.execute(sql)
                print("✅ Таблица/индекс созданы")
            except Exception as e:
                print(f"❌ Ошибка: {e}")

        self.conn.commit()

    def import_to_simple_db(self, metadata_df):
        """Импорт в упрощенную базу данных"""
        print("\n📥 ИМПОРТ В БАЗУ ДАННЫХ")
        print("=" * 50)

        imported = 0
        for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Импорт"):
            try:
                self.conn.execute('''
                    INSERT OR IGNORE INTO photos_enhanced 
                    (photo_id, object_type, month, file_path, has_image)
                    VALUES (?, ?, ?, ?, ?)
                ''', (row['photo_id'], row['object_type'], row['month'],
                      row['expected_path'], row['has_image']))
                imported += 1
            except Exception as e:
                print(f"⚠️ Ошибка импорта {row['photo_id']}: {e}")

        self.conn.commit()
        print(f"✅ Импортировано {imported} записей")

    def get_database_stats(self):
        """Статистика базы данных"""
        print("\n📊 СТАТИСТИКА БАЗЫ ДАННЫХ")
        print("=" * 50)

        queries = [
            ("Всего записей", "SELECT COUNT(*) FROM photos_enhanced"),
            ("Записи с изображениями", "SELECT COUNT(*) FROM photos_enhanced WHERE has_image = 1"),
            ("По типам объектов", "SELECT object_type, COUNT(*) FROM photos_enhanced GROUP BY object_type"),
            ("По месяцам", "SELECT month, COUNT(*) FROM photos_enhanced GROUP BY month")
        ]

        for name, query in queries:
            result = self.conn.execute(query).fetchone()[0]
            print(f"   {name}: {result}")


def check_system_limits():
    """Проверка системных ограничений"""
    print("🔍 ПРОВЕРКА СИСТЕМНЫХ ОГРАНИЧЕНИЙ WINDOWS")
    print("=" * 50)

    # Максимальная длина пути в Windows
    max_path = 260
    print(f"📏 Максимальная длина пути Windows: {max_path} символов")

    # Проверяем базовый путь
    base_path = r"C:\PythonProjects\hahaton_vswm_obji_tresh\Метаданные\Метаданные\Выгрузка"
    print(f"📁 Длина базового пути: {len(base_path)} символов")

    if len(base_path) > 100:
        print("⚠️ Базовый путь довольно длинный, используем короткие имена")


def main():
    """Основная функция с обработкой ошибок Windows"""
    base_path = r"C:\PythonProjects\hahaton_vswm_obji_tresh\Метаданные\Метаданные\Выгрузка"

    print("🚀 WINDOWS-СОВМЕСТИМАЯ ОРГАНИЗАЦИЯ ДАННЫХ")
    print("=" * 60)

    # Проверяем ограничения
    check_system_limits()

    try:
        # Инициализируем организатор
        organizer = WindowsDataOrganizer(base_path)

        # 1. Распаковываем архивы
        organizer.unpack_and_organize_archives()

        # 2. Обрабатываем метаданные
        metadata_df = organizer.process_json_metadata_simple()

        # 3. Работа с базой данных
        db_manager = SimpleDatabaseManager(organizer.db_path)
        db_manager.create_simple_schema()
        db_manager.import_to_simple_db(metadata_df)
        db_manager.get_database_stats()

        print("\n🎉 ДАННЫЕ УСПЕШНО ОРГАНИЗОВАНЫ!")
        print("📁 Структура создана с учетом ограничений Windows")

    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n💡 СОВЕТ: Для работы с большими объемами данных рассмотрите:")
        print("   - Использование Linux-сервера")
        print("   - Включение длинных путей в Windows (если возможно)")
        print("   - Работу с данными в корне диска (C:\\data)")


if __name__ == "__main__":
    main()