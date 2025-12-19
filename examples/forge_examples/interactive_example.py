"""
Пример использования Interactive Model Manager
===============================================

Модуль interactive предоставляет полноценную интерактивную консоль
для управления моделями, валидации датасетов и настройки fine-tuning.

Запуск:
    python examples/forge_examples/interactive_example.py

Требования:
    - Директория с моделями (HuggingFace format)
    - Директория с датасетами (JSONL format)
"""

import os
import sys

# Добавляем путь к transformers
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def example_scan_models():
    """Пример сканирования моделей."""
    from transformers.interactive import InteractiveModelManager
    
    print("=" * 60)
    print("ПРИМЕР 1: Сканирование моделей")
    print("=" * 60)
    
    # Создаём менеджер с путями к моделям и датасетам
    manager = InteractiveModelManager(
        models_dir="./models",  # Укажите свой путь
        datasets_dir="./datasets"
    )
    
    # Сканируем модели
    models = manager.scan_models()
    
    if models:
        print(f"\n✅ Найдено {len(models)} моделей:\n")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model.name}")
            print(f"     Путь: {model.path}")
            print(f"     Размер: {model.size_gb:.2f} GB")
            if model.num_parameters:
                print(f"     Параметры: {model.num_parameters}")
            print(f"     Safetensors: {'✓' if model.has_safetensors else '✗'}")
            print(f"     Tokenizer: {'✓' if model.has_tokenizer else '✗'}")
            print()
    else:
        print("\n⚠️ Модели не найдены в ./models")
        print("   Укажите директорию с HuggingFace моделями")


def example_scan_datasets():
    """Пример сканирования датасетов."""
    from transformers.interactive import InteractiveModelManager
    
    print("=" * 60)
    print("ПРИМЕР 2: Сканирование датасетов")
    print("=" * 60)
    
    manager = InteractiveModelManager(
        models_dir="./models",
        datasets_dir="./datasets"
    )
    
    datasets = manager.scan_datasets()
    
    if datasets:
        print(f"\n✅ Найдено {len(datasets)} датасетов:\n")
        for i, ds in enumerate(datasets, 1):
            print(f"  {i}. {ds.name}")
            print(f"     Путь: {ds.path}")
            print(f"     Размер: {ds.size_mb:.2f} MB")
            print(f"     Строк: {ds.num_lines}")
            print(f"     Формат: {ds.format}")
            print()
    else:
        print("\n⚠️ Датасеты не найдены в ./datasets")
        print("   Укажите директорию с .jsonl файлами")


def example_validate_dataset():
    """Пример валидации датасета."""
    from transformers.interactive import InteractiveModelManager
    import tempfile
    import json
    
    print("=" * 60)
    print("ПРИМЕР 3: Валидация датасета")
    print("=" * 60)
    
    # Создаём тестовый датасет
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        # Правильная строка
        valid_line = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi! How can I help you?"}
            ]
        }
        f.write(json.dumps(valid_line) + "\n")
        
        # Ещё одна правильная строка
        valid_line2 = {
            "messages": [
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a programming language."}
            ]
        }
        f.write(json.dumps(valid_line2) + "\n")
        
        temp_path = f.name
    
    manager = InteractiveModelManager()
    
    print(f"\n📋 Валидируем: {temp_path}\n")
    
    result = manager.validate_dataset(temp_path)
    
    print(f"  Валиден: {'✅ Да' if result.valid else '❌ Нет'}")
    print(f"  Всего строк: {result.total_lines}")
    print(f"  Валидных: {result.valid_lines}")
    
    if result.errors:
        print("\n  ❌ Ошибки:")
        for error in result.errors:
            print(f"     - {error}")
    
    if result.warnings:
        print("\n  ⚠️ Предупреждения:")
        for warning in result.warnings:
            print(f"     - {warning}")
    
    # Удаляем временный файл
    os.unlink(temp_path)
    print("\n  ✅ Тестовый датасет удалён")


def example_show_format():
    """Показать формат датасета."""
    from transformers.interactive import InteractiveModelManager
    
    print("=" * 60)
    print("ПРИМЕР 4: Формат датасета ChatML")
    print("=" * 60)
    
    manager = InteractiveModelManager()
    manager.show_dataset_example()


def example_run_interactive():
    """Запустить полную интерактивную сессию."""
    from transformers.interactive import InteractiveModelManager
    
    print("=" * 60)
    print("ПРИМЕР 5: Интерактивная сессия")
    print("=" * 60)
    print("\n⚠️ Это запустит интерактивный режим.")
    print("   Для выхода введите 'q' или нажмите Ctrl+C")
    
    response = input("\nЗапустить? (y/n): ")
    
    if response.lower() == 'y':
        manager = InteractiveModelManager(
            models_dir="./models",
            datasets_dir="./datasets"
        )
        manager.run()
    else:
        print("Пропущено.")


def main():
    """Главная функция."""
    print("\n" + "=" * 60)
    print("  INTERACTIVE MODEL MANAGER - ПРИМЕРЫ")
    print("=" * 60 + "\n")
    
    print("Выберите пример:")
    print("  1. Сканирование моделей")
    print("  2. Сканирование датасетов")
    print("  3. Валидация датасета")
    print("  4. Показать формат ChatML")
    print("  5. Интерактивная сессия (полный режим)")
    print("  0. Выход")
    print()
    
    choice = input("Выбор (0-5): ")
    
    if choice == "1":
        example_scan_models()
    elif choice == "2":
        example_scan_datasets()
    elif choice == "3":
        example_validate_dataset()
    elif choice == "4":
        example_show_format()
    elif choice == "5":
        example_run_interactive()
    elif choice == "0":
        print("Выход.")
    else:
        print("Неверный выбор.")


if __name__ == "__main__":
    main()
