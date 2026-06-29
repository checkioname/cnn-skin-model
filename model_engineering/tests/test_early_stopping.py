import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from application.callbacks.EarlyStopping import EarlyStopping


class TestEarlyStopping:
    def test_initialization(self):
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        assert early_stopping.patience == 3
        assert early_stopping.min_delta == 0.01
        assert early_stopping.best_loss is None
        assert early_stopping.counter == 0
        assert early_stopping.early_stop is False

    def test_does_not_stop_when_improving(self):
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        early_stopping(0.5)
        assert early_stopping.early_stop is False
        early_stopping(0.4)
        assert early_stopping.early_stop is False
        early_stopping(0.3)
        assert early_stopping.early_stop is False
        assert early_stopping.counter == 0

    def test_stops_after_patience(self):
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        early_stopping(0.5)
        assert early_stopping.early_stop is False
        early_stopping(0.52)
        assert early_stopping.counter == 1
        assert early_stopping.early_stop is False
        early_stopping(0.54)
        assert early_stopping.counter == 2
        assert early_stopping.early_stop is False
        early_stopping(0.55)
        assert early_stopping.counter == 3
        assert early_stopping.early_stop is False
        early_stopping(0.56)
        assert early_stopping.early_stop is True

    def test_resets_counter_on_improvement(self):
        early_stopping = EarlyStopping(patience=3, min_delta=0.01)
        early_stopping(0.5)
        early_stopping(0.52)
        assert early_stopping.counter == 1
        early_stopping(0.49)
        assert early_stopping.counter == 0
        assert early_stopping.best_loss == 0.49
