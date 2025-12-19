"""
Демо Interactive Model Manager v1.0.9
=====================================

Запуск:
    python demo_interactive.py
"""
import sys
sys.path.insert(0, 'src')

from transformers.interactive import InteractiveModelManager

print("=" * 70)
print("🧪 ДЕМО: Interactive Model Manager v1.0.9")
print("=" * 70)
print()

manager = InteractiveModelManager(
    models_dir="./demo_models",
    datasets_dir="./demo_datasets"
)

manager.run()
