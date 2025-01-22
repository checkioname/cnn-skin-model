from application.callbacks.EarlyStopping import EarlyStopping
import pytest

def test_initialization():
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)
    
    assert early_stopping.patience == 3
    assert early_stopping.min_delta == 0.01
    assert early_stopping.best_loss is None
    assert early_stopping.counter == 0
    assert early_stopping.early_stop is False



def test_no_improvement_increments_counter():
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)

    #primeira iteracao
    early_stopping(0.5)

    #segunda iteracao
    early_stopping(0.52)
    assert early_stopping.counter == 1
    assert early_stopping.early_stop is False

    #terceira iteracao
    early_stopping(0.54)
    assert early_stopping.counter == 2
    assert early_stopping.early_stop is False

    # Deve aumentar o contador ja que a loss atual é maior que a loss anterior mais o delta
    early_stopping(0.55)
    assert early_stopping.counter == 3
    assert early_stopping.early_stop is False