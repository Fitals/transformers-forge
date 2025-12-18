"""
🔬 EMA Benchmark — Практическое доказательство (v2)
====================================================

Этот скрипт демонстрирует эффект EMA на реальном обучении.

МЕТОДОЛОГИЯ:
- Обучаем ОДНУ модель с EMA
- Сравниваем normal weights vs EMA weights ОДНОЙ модели
- Это показывает реальную пользу EMA

Запуск:
    python benchmarks/ema_benchmark.py

Примерное время: 30-60 секунд на CPU
"""

import time
import sys
import os

# Добавляем путь к src для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def run_benchmark():
    """Запуск benchmark сравнения EMA vs Normal weights."""
    
    print("=" * 60)
    print("🔬 EMA BENCHMARK v2 — Transformers Forge")
    print("=" * 60)
    print()
    print("📋 МЕТОДОЛОГИЯ:")
    print("   Обучаем ОДНУ модель с EMA и сравниваем:")
    print("   - Normal weights (финальные веса после обучения)")
    print("   - EMA weights (усреднённые веса)")
    print()
    
    # Импорты
    print("📦 Загрузка библиотек...")
    
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("❌ PyTorch не установлен. Установите: pip install torch")
        return
    
    try:
        from transformers.ema import create_ema_state, update_ema_state, apply_ema_state
    except ImportError:
        print("❌ Transformers Forge не установлен. Установите: pip install -e .")
        return
    
    print("✅ Библиотеки загружены")
    print()
    
    # Конфигурация
    HIDDEN_SIZE = 256
    NUM_LAYERS = 4
    BATCH_SIZE = 32
    NUM_SAMPLES = 2000
    NUM_STEPS = 300
    LEARNING_RATE = 2e-3  # Высокий LR для создания шума
    EMA_DECAY = 0.99
    
    print("⚙️ Конфигурация:")
    print(f"   Hidden size: {HIDDEN_SIZE}")
    print(f"   Layers: {NUM_LAYERS}")
    print(f"   Batch size: {BATCH_SIZE}")
    print(f"   Training steps: {NUM_STEPS}")
    print(f"   Learning rate: {LEARNING_RATE} (высокий для шума)")
    print(f"   EMA decay: {EMA_DECAY}")
    print()
    
    # Создаём простую модель
    class SimpleTransformer(nn.Module):
        def __init__(self, hidden_size, num_layers):
            super().__init__()
            self.embedding = nn.Linear(hidden_size, hidden_size)
            self.layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size * 4),
                    nn.GELU(),
                    nn.Linear(hidden_size * 4, hidden_size),
                    nn.LayerNorm(hidden_size)
                )
                for _ in range(num_layers)
            ])
            self.head = nn.Linear(hidden_size, hidden_size)
        
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = x + layer(x)
            return self.head(x)
    
    # Генерируем синтетические данные с шумом
    print("📊 Генерация данных...")
    torch.manual_seed(42)
    X = torch.randn(NUM_SAMPLES, HIDDEN_SIZE)
    # Target с шумом (симуляция реальных данных)
    Y = torch.sin(X) * 0.5 + torch.randn_like(X) * 0.2  # Больше шума
    
    dataset = TensorDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Evaluation set (чистый, без шума в target)
    X_eval = torch.randn(500, HIDDEN_SIZE)
    Y_eval = torch.sin(X_eval) * 0.5  # Без шума — истинная функция
    
    def evaluate(model, X_eval, Y_eval):
        """Оценка модели."""
        model.eval()
        with torch.no_grad():
            pred = model(X_eval)
            loss = nn.MSELoss()(pred, Y_eval)
        model.train()
        return loss.item()
    
    # ========================================================================
    # Обучение с EMA
    # ========================================================================
    print()
    print("-" * 60)
    print("🟢 Обучение модели с EMA tracking")
    print("-" * 60)
    
    model = SimpleTransformer(HIDDEN_SIZE, NUM_LAYERS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
    # Инициализируем EMA
    ema_state = create_ema_state(model)
    
    eval_normal_history = []
    eval_ema_history = []
    
    start_time = time.time()
    step = 0
    
    for epoch in range(20):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            
            # Обновляем EMA
            update_ema_state(model, ema_state, decay=EMA_DECAY)
            
            if step % 30 == 0:
                # Eval с normal весами
                eval_normal = evaluate(model, X_eval, Y_eval)
                eval_normal_history.append(eval_normal)
                
                # Eval с EMA весами
                backup = apply_ema_state(model, ema_state)
                eval_ema = evaluate(model, X_eval, Y_eval)
                eval_ema_history.append(eval_ema)
                apply_ema_state(model, backup)
                
                diff = ((eval_normal - eval_ema) / eval_normal) * 100 if eval_normal > 0 else 0
                marker = "✅" if eval_ema < eval_normal else "⚠️"
                
                print(f"   Step {step:3d} | Normal: {eval_normal:.4f} | EMA: {eval_ema:.4f} | {marker} {diff:+.1f}%")
            
            step += 1
            if step >= NUM_STEPS:
                break
        if step >= NUM_STEPS:
            break
    
    training_time = time.time() - start_time
    
    # ========================================================================
    # Финальное сравнение
    # ========================================================================
    print()
    print("-" * 60)
    print("📊 ФИНАЛЬНОЕ СРАВНЕНИЕ")
    print("-" * 60)
    
    # Финальный eval с normal весами
    final_normal = evaluate(model, X_eval, Y_eval)
    
    # Применяем EMA и eval
    backup = apply_ema_state(model, ema_state)
    final_ema = evaluate(model, X_eval, Y_eval)
    
    improvement = ((final_normal - final_ema) / final_normal) * 100
    
    print()
    print(f"   {'Метрика':<25} {'Normal':<15} {'EMA':<15} {'Разница':<15}")
    print(f"   {'-'*25} {'-'*15} {'-'*15} {'-'*15}")
    print(f"   {'Final Eval Loss':<25} {final_normal:<15.4f} {final_ema:<15.4f} {improvement:+.1f}%")
    print()
    
    if improvement > 0:
        print(f"   ✅ EMA улучшил качество на {improvement:.1f}%!")
        print(f"   📌 EMA веса лучше чем normal веса ОДНОЙ модели")
    else:
        print(f"   ⚠️ EMA не показал улучшения ({improvement:.1f}%)")
    
    # ========================================================================
    # Анализ истории
    # ========================================================================
    print()
    print("-" * 60)
    print("📈 АНАЛИЗ ИСТОРИИ")
    print("-" * 60)
    
    ema_wins = sum(1 for n, e in zip(eval_normal_history, eval_ema_history) if e < n)
    total_evals = len(eval_normal_history)
    
    print()
    print(f"   EMA лучше в {ema_wins}/{total_evals} точках ({100*ema_wins/total_evals:.0f}%)")
    print()
    
    # Последние 5 измерений
    print("   Последние 5 измерений:")
    for i, (n, e) in enumerate(zip(eval_normal_history[-5:], eval_ema_history[-5:])):
        marker = "✅" if e < n else "⚠️"
        print(f"      {marker} Normal: {n:.4f}, EMA: {e:.4f}")
    
    # ========================================================================
    # Выводы
    # ========================================================================
    print()
    print("=" * 60)
    print("📝 ВЫВОДЫ")
    print("=" * 60)
    print("""
   КЛЮЧЕВОЙ ИНСАЙТ:
   
   EMA сглаживает колебания весов вызванные:
   - Шумом в данных
   - Высоким learning rate
   - Стохастичностью SGD
   
   КОГДА EMA ПОМОГАЕТ:
   ✅ Шумные данные (реальные датасеты)
   ✅ Высокий learning rate
   ✅ Длинное обучение (накопление истории)
   ✅ Большие модели (больше variance)
   
   КОГДА EMA НЕ ПОМОГАЕТ:
   ❌ Чистые синтетические данные
   ❌ Очень маленькие модели
   ❌ Короткое обучение
""")
    print("=" * 60)
    
    return {
        "final_normal": final_normal,
        "final_ema": final_ema,
        "improvement_percent": improvement,
        "ema_win_rate": ema_wins / total_evals,
        "training_time": training_time,
    }


if __name__ == "__main__":
    results = run_benchmark()
