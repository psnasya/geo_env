# 1.2_deep_analysis.py для структуры
import os
import json
import zipfile
import pandas as pd
from pathlib import Path
import re


def deep_analyze_data():
    """Глубокий анализ данных с исправленным парсингом"""
    base_path = r"C:\PythonProjects\hahaton_vswm_obji_tresh\Метаданные\Метаданные\Выгрузка"

    print("🔍 ГЛУБОКИЙ АНАЛИЗ ДАННЫХ")
    print("=" * 50)

    # Улучшенный парсинг имен файлов
    def parse_filename(filename):
        """Парсим имя файла для извлечения структуры"""
        # Убираем расширение
        name_without_ext = Path(filename).stem

        # Определяем тип объекта
        if 'не соответствующие' in name_without_ext or 'несоответствующие' in name_without_ext:
            obj_type = "non_compliant_buildings"
        elif 'строительная площадка' in name_without_ext.lower():
            obj_type = "construction_site"
        else:
            obj_type = "unknown"

        # Ищем код объекта (00-022, 18-001)
        code_match = re.search(r'(\d{2}-\d{3})', name_without_ext)
        obj_code = code_match.group(1) if code_match else "unknown"

        # Ищем месяц
        months = ['январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
                  'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь']
        month = "unknown"
        for m in months:
            if m in name_without_ext.lower():
                month = m
                break

        return obj_type, obj_code, month

    # Собираем детальную информацию
    detailed_analysis = []

    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)

            obj_type, obj_code, month = parse_filename(file)
            file_ext = Path(file).suffix.lower()
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

            detailed_analysis.append({
                'filename': file,
                'full_path': file_path,
                'object_type': obj_type,
                'object_code': obj_code,
                'month': month,
                'extension': file_ext,
                'size_mb': round(file_size_mb, 2),
                'file_category': 'archive' if file_ext == '.zip' else
                'metadata' if file_ext == '.json' else
                'additional' if file_ext == '.xlsx' else 'other'
            })

    df = pd.DataFrame(detailed_analysis)

    print("📊 ДЕТАЛЬНАЯ СТАТИСТИКА:")
    print(f"Всего файлов: {len(df)}")
    print(f"Общий размер: {df['size_mb'].sum():.2f} MB")

    # Группировка по типам объектов
    print("\n🏗️ РАСПРЕДЕЛЕНИЕ ПО ТИПАМ ОБЪЕКТОВ:")
    obj_stats = df.groupby('object_type').agg({
        'filename': 'count',
        'size_mb': 'sum'
    }).round(2)
    print(obj_stats)

    return df


def extract_detailed_json_metadata():
    """Детальный анализ JSON метаданных"""
    print("\n🔍 ДЕТАЛЬНЫЙ АНАЛИЗ JSON МЕТАДАННЫХ")
    print("=" * 50)

    base_path = r"C:\PythonProjects\hahaton_vswm_obji_tresh\Метаданные\Метаданные\Выгрузка"
    json_files = list(Path(base_path).rglob('*.json'))

    metadata_details = []

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print(f"\n📄 Файл: {json_file.name}")
            print(
                f"📁 Тип объекта: {'non_compliant_buildings' if 'не соответствующие' in json_file.name else 'construction_site'}")

            if isinstance(data, dict):
                print("🔑 Ключи JSON:")
                for key, value in data.items():
                    if isinstance(value, list):
                        print(f"   {key}: список из {len(value)} элементов")
                        # Анализ первых элементов если это список
                        if value and isinstance(value[0], dict):
                            print(f"     Пример ключей элемента: {list(value[0].keys())[:5]}")
                    else:
                        print(f"   {key}: {str(value)[:100]}...")

            elif isinstance(data, list):
                print(f"📋 JSON содержит список из {len(data)} элементов")
                if data and isinstance(data[0], dict):
                    print(f"🔑 Ключи элементов: {list(data[0].keys())}")

            metadata_details.append({
                'file': json_file.name,
                'data_type': type(data).__name__,
                'size': len(data) if isinstance(data, list) else len(data.keys()) if isinstance(data, dict) else 1,
                'sample': data[0] if isinstance(data, list) and data else list(data.keys())[:3] if isinstance(data,
                                                                                                              dict) else data
            })

        except Exception as e:
            print(f"❌ Ошибка анализа {json_file.name}: {e}")

    return metadata_details


def analyze_zip_contents():
    """Детальный анализ содержимого ZIP архивов"""
    print("\n📦 ДЕТАЛЬНЫЙ АНАЛИЗ ZIP АРХИВОВ")
    print("=" * 50)

    base_path = r"C:\PythonProjects\hahaton_vswm_obji_tresh\Метаданные\Метаданные\Выгрузка"
    zip_files = list(Path(base_path).rglob('*.zip'))

    zip_analysis = []

    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                all_files = zip_ref.namelist()
                image_files = [f for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                other_files = [f for f in all_files if not f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                # Анализ структуры папок внутри архива
                folders = set()
                for file in all_files:
                    folder = os.path.dirname(file)
                    if folder:
                        folders.add(folder)

                print(f"\n📁 Архив: {zip_file.name}")
                print(f"   📊 Всего файлов: {len(all_files)}")
                print(f"   🖼️  Изображений: {len(image_files)}")
                print(f"   📄 Других файлов: {len(other_files)}")
                print(f"   📂 Папок внутри: {len(folders)}")

                if folders:
                    print(f"   Структура папок: {list(folders)[:3]}...")

                # Анализ имен изображений
                if image_files:
                    sample_images = image_files[:3]
                    print(f"   Примеры изображений:")
                    for img in sample_images:
                        print(f"     - {os.path.basename(img)}")

                zip_analysis.append({
                    'archive_name': zip_file.name,
                    'total_files': len(all_files),
                    'image_count': len(image_files),
                    'folder_count': len(folders),
                    'folder_sample': list(folders)[:3] if folders else [],
                    'image_sample': [os.path.basename(img) for img in image_files[:2]]
                })

        except Exception as e:
            print(f"❌ Ошибка анализа архива {zip_file.name}: {e}")

    return zip_analysis


def extract_sample_images_from_zip():
    """Извлечение примеров изображений для анализа"""
    print("\n🖼️  ИЗВЛЕЧЕНИЕ ПРИМЕРОВ ИЗОБРАЖЕНИЙ")
    print("=" * 50)

    base_path = r"C:\PythonProjects\hahaton_vswm_obji_tresh\Метаданные\Метаданные\Выгрузка"
    zip_files = list(Path(base_path).rglob('*.zip'))

    extract_path = r"C:\PythonProjects\hahaton_vswm_obji_tresh\sample_images"
    os.makedirs(extract_path, exist_ok=True)

    for zip_file in zip_files[:2]:  # Берем первые 2 архива для примера
        try:
            print(f"\n📦 Извлекаем примеры из: {zip_file.name}")

            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                image_files = [f for f in zip_ref.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                # Извлекаем 3 случайных изображения
                sample_images = image_files[:3]
                for img_path in sample_images:
                    try:
                        # Извлекаем в подпапку по имени архива
                        archive_name = Path(zip_file).stem
                        target_dir = os.path.join(extract_path, archive_name)
                        os.makedirs(target_dir, exist_ok=True)

                        # Извлекаем файл
                        zip_ref.extract(img_path, target_dir)
                        extracted_path = os.path.join(target_dir, img_path)

                        print(f"   ✅ Извлечено: {os.path.basename(img_path)}")

                    except Exception as e:
                        print(f"   ❌ Ошибка извлечения {img_path}: {e}")

        except Exception as e:
            print(f"❌ Ошибка работы с архивом {zip_file.name}: {e}")


def generate_integration_plan(df, metadata_details, zip_analysis):
    """Генерация плана интеграции с существующей системой"""
    print("\n📋 ПЛАН ИНТЕГРАЦИИ С СУЩЕСТВУЮЩЕЙ СИСТЕМОЙ")
    print("=" * 60)

    total_images = sum([z['image_count'] for z in zip_analysis])

    print(f"🎯 ОБЩАЯ СТАТИСТИКА ДЛЯ ИНТЕГРАЦИИ:")
    print(f"   • Всего архивов: {len(zip_analysis)}")
    print(f"   • Всего изображений: {total_images}")
    print(
        f"   • Объекты недвижимости: {len([z for z in zip_analysis if 'не соответствующие' in z['archive_name']])} архивов")
    print(
        f"   • Строительные площадки: {len([z for z in zip_analysis if 'строительная' in z['archive_name'].lower()])} архивов")

    print(f"\n🚀 ЭТАПЫ ИНТЕГРАЦИИ:")
    print(f"1. 📦 Распаковка архивов в структурированные папки")
    print(f"2. 🔗 Связывание изображений с метаданными из JSON")
    print(f"3. 🗃️  Расширение базы данных новыми таблицами")
    print(f"4. 🤖 Дообучение нейросети на новых типах объектов")
    print(f"5. 🌐 Интеграция с веб-интерфейсом")

    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    print(f"   • Создать отдельные папки для каждого типа объектов")
    print(f"   • Использовать коды объектов (00-022, 18-001) для организации")
    print(f"   • Добавить поля для месяца и типа нарушения в базу данных")
    print(f"   • Рассмотреть возможность сегментации изображений по типам объектов")


def main():
    """Основная функция глубокого анализа"""
    print("🚀 ЗАПУСК ГЛУБОКОГО АНАЛИЗА ДАННЫХ ХАКАТОНА")
    print("=" * 60)

    # 1. Детальный анализ структуры
    df = deep_analyze_data()

    # 2. Анализ JSON метаданных
    metadata_details = extract_detailed_json_metadata()

    # 3. Анализ ZIP архивов
    zip_analysis = analyze_zip_contents()

    # 4. Извлечение примеров изображений (опционально)
    extract_sample_images_from_zip()

    # 5. Генерация плана интеграции
    generate_integration_plan(df, metadata_details, zip_analysis)

    # 6. Сохранение результатов
    output_file = r"C:\PythonProjects\hahaton_vswm_obji_tresh\deep_analysis_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("ОТЧЕТ ПО ГЛУБОКОМУ АНАЛИЗУ ДАННЫХ\n")
        f.write("=" * 50 + "\n")
        f.write(f"Всего архивов: {len(zip_analysis)}\n")
        f.write(f"Всего изображений: {sum([z['image_count'] for z in zip_analysis])}\n")
        f.write(f"Общий размер: {df['size_mb'].sum():.2f} MB\n")

    print(f"\n💾 Отчет сохранен: {output_file}")
    print("\n✅ Анализ завершен! Готовимся к интеграции.")


if __name__ == "__main__":
    main()