"""
Тесты для модуля Dataset Utils
==============================

Запуск:
    pytest tests/forge/test_dataset_utils.py -v
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import List, Dict, Any


class MockDataset:
    """Mock dataset for testing"""
    
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
        self.column_names = list(data[0].keys()) if data else []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class MockTokenizer:
    """Mock tokenizer for testing"""
    
    def __init__(self, tokens_per_word: float = 1.5):
        self.tokens_per_word = tokens_per_word
    
    def __call__(self, text, truncation=False, add_special_tokens=True):
        words = text.split()
        num_tokens = int(len(words) * self.tokens_per_word)
        return {"input_ids": [0] * num_tokens}


class TestAnalyzeDataset:
    """Тесты для analyze_dataset"""
    
    @pytest.fixture
    def sample_dataset(self):
        return MockDataset([
            {"text": "Hello world this is a test"},
            {"text": "Another example with more words here"},
            {"text": "Short text"},
            {"text": "This is a longer piece of text with many more words in it"},
        ])
    
    @pytest.fixture
    def chat_dataset(self):
        return MockDataset([
            {"messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there! How can I help?"}
            ]},
            {"messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is artificial intelligence"}
            ]},
        ])
    
    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()
    
    def test_analyze_basic_dataset(self, sample_dataset, tokenizer):
        """Тест базового анализа датасета"""
        from transformers.dataset_utils import analyze_dataset
        
        stats = analyze_dataset(sample_dataset, tokenizer)
        
        assert stats.num_samples == 4
        assert stats.num_columns == 1
        assert "text" in stats.column_names
        assert stats.total_tokens > 0
        assert stats.mean_tokens > 0
    
    def test_analyze_without_tokenizer(self, sample_dataset):
        """Тест анализа без токенизатора"""
        from transformers.dataset_utils import analyze_dataset
        
        stats = analyze_dataset(sample_dataset, tokenizer=None)
        
        assert stats.num_samples == 4
        # Должен использовать эвристику words * 1.3
        assert stats.total_tokens > 0
    
    def test_analyze_chat_format(self, chat_dataset, tokenizer):
        """Тест анализа chat формата"""
        from transformers.dataset_utils import analyze_dataset
        
        stats = analyze_dataset(chat_dataset, tokenizer, text_column="messages")
        
        assert stats.num_samples == 2
        assert stats.total_tokens > 0
    
    def test_stats_has_recommendations(self, sample_dataset, tokenizer):
        """Тест что статистика содержит рекомендации"""
        from transformers.dataset_utils import analyze_dataset
        
        stats = analyze_dataset(sample_dataset, tokenizer)
        
        assert stats.recommended_batch_size > 0
        assert stats.recommended_max_length > 0
    
    def test_length_distribution(self, sample_dataset, tokenizer):
        """Тест распределения длин"""
        from transformers.dataset_utils import analyze_dataset
        
        stats = analyze_dataset(sample_dataset, tokenizer)
        
        assert isinstance(stats.length_distribution, dict)
        assert sum(stats.length_distribution.values()) > 0


class TestEstimateTokens:
    """Тесты для estimate_tokens"""
    
    @pytest.fixture
    def sample_dataset(self):
        return MockDataset([
            {"text": "Hello world"},
            {"text": "Another example"},
        ])
    
    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()
    
    def test_estimate_tokens_returns_int(self, sample_dataset, tokenizer):
        """Тест что estimate_tokens возвращает число"""
        from transformers.dataset_utils import estimate_tokens
        
        tokens = estimate_tokens(sample_dataset, tokenizer)
        
        assert isinstance(tokens, int)
        assert tokens > 0


class TestRecommendBatchSize:
    """Тесты для recommend_batch_size"""
    
    def test_recommend_with_stats(self):
        """Тест рекомендаций на основе статистики"""
        from transformers.dataset_utils import recommend_batch_size, DatasetStats
        
        stats = DatasetStats(mean_tokens=256.0)
        
        rec = recommend_batch_size(dataset_stats=stats)
        
        assert "batch_size" in rec
        assert "gradient_accumulation" in rec
        assert "effective_batch" in rec
        assert rec["batch_size"] > 0
    
    def test_recommend_short_sequences(self):
        """Тест рекомендаций для коротких последовательностей"""
        from transformers.dataset_utils import recommend_batch_size, DatasetStats
        
        stats = DatasetStats(mean_tokens=64.0)
        
        rec = recommend_batch_size(dataset_stats=stats)
        
        # Короткие последовательности → большой batch
        assert rec["batch_size"] >= 8
    
    def test_recommend_long_sequences(self):
        """Тест рекомендаций для длинных последовательностей"""
        from transformers.dataset_utils import recommend_batch_size, DatasetStats
        
        stats = DatasetStats(mean_tokens=2048.0)
        
        rec = recommend_batch_size(dataset_stats=stats)
        
        # Длинные последовательности → маленький batch
        assert rec["batch_size"] <= 2
    
    def test_recommend_with_memory_constraint(self):
        """Тест рекомендаций с ограничением памяти"""
        from transformers.dataset_utils import recommend_batch_size, DatasetStats
        
        stats = DatasetStats(mean_tokens=256.0)
        
        rec = recommend_batch_size(
            dataset_stats=stats,
            available_memory_gb=4.0  # Мало памяти
        )
        
        # Должен уменьшить batch
        assert "Limited GPU memory" in rec["reason"]


class TestDatasetAnalyzer:
    """Тесты для DatasetAnalyzer"""
    
    @pytest.fixture
    def sample_dataset(self):
        return MockDataset([
            {"text": "Hello world this is a test"},
            {"text": "Another example with more words here"},
            {"text": "Short text"},
        ])
    
    @pytest.fixture
    def tokenizer(self):
        return MockTokenizer()
    
    def test_analyzer_initialization(self, sample_dataset, tokenizer):
        """Тест инициализации анализатора"""
        from transformers.dataset_utils import DatasetAnalyzer
        
        analyzer = DatasetAnalyzer(sample_dataset, tokenizer)
        
        assert analyzer.dataset is sample_dataset
        assert analyzer.tokenizer is tokenizer
    
    def test_analyzer_analyze(self, sample_dataset, tokenizer):
        """Тест метода analyze"""
        from transformers.dataset_utils import DatasetAnalyzer
        
        analyzer = DatasetAnalyzer(sample_dataset, tokenizer)
        stats = analyzer.analyze()
        
        assert stats is not None
        assert stats.num_samples == 3
    
    def test_analyzer_get_recommendations(self, sample_dataset, tokenizer):
        """Тест получения рекомендаций"""
        from transformers.dataset_utils import DatasetAnalyzer
        
        analyzer = DatasetAnalyzer(sample_dataset, tokenizer)
        analyzer.analyze()
        
        rec = analyzer.get_recommendations()
        
        assert "batch_size" in rec
        assert "max_length" in rec
        assert "total_tokens" in rec
    
    def test_analyzer_print_report(self, sample_dataset, tokenizer, capsys):
        """Тест печати отчёта"""
        from transformers.dataset_utils import DatasetAnalyzer
        
        analyzer = DatasetAnalyzer(sample_dataset, tokenizer)
        analyzer.print_report()
        
        captured = capsys.readouterr()
        assert "DATASET ANALYSIS REPORT" in captured.out
        assert "Token Statistics" in captured.out


class TestDatasetStats:
    """Тесты для DatasetStats dataclass"""
    
    def test_dataclass_creation(self):
        """Тест создания dataclass"""
        from transformers.dataset_utils import DatasetStats
        
        stats = DatasetStats(
            num_samples=100,
            total_tokens=10000,
            mean_tokens=100.0,
        )
        
        assert stats.num_samples == 100
        assert stats.total_tokens == 10000
        assert stats.mean_tokens == 100.0
    
    def test_dataclass_repr(self):
        """Тест repr метода"""
        from transformers.dataset_utils import DatasetStats
        
        stats = DatasetStats(
            num_samples=100,
            total_tokens=10000,
            mean_tokens=100.0,
        )
        
        repr_str = repr(stats)
        assert "samples=100" in repr_str
        assert "tokens=10,000" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
