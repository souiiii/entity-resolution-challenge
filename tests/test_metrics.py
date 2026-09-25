from src.evaluation.metrics import f05


def test_f05_perfect():
    assert f05(1.0, 1.0) == 1.0


def test_f05_zero():
    assert f05(0.0, 0.0) == 0.0


def test_f05_precision_weighting():
    assert f05(0.9, 0.5) > f05(0.5, 0.9)
