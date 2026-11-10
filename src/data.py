from typing import Iterable

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from ._config import CHARS_BY_BRIGHTNESS_ASC


def to_one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Превращает вектор меток размера (N,) в матрицу (N, num_classes)."""
    return np.eye(num_classes)[labels]


def print_ascii_digit(image_flat: np.ndarray, label: int,
                      chars: str = CHARS_BY_BRIGHTNESS_ASC) -> None:
    grid = image_flat.reshape(8, 8)

    for row in grid:
        line = "".join(chars[int(val * (len(chars) - 1))] * 2 for val in row)
        print(line)


def get_data(test_size: float = 0.2, random_state: int = 42):
    digits = load_digits()

    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int64)

    X /= 16.0

    print(y)

    Y_one_hot = to_one_hot(y)

    print(Y_one_hot)

    X_train, X_test, y_train, y_test, Y_train_hot, Y_test_hot = train_test_split(
        X, y, Y_one_hot,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test, Y_train_hot, Y_test_hot


def test():
    X_train, X_test, y_train, y_test, Y_train_hot, Y_test_hot = get_data()

    print(f"Обучающая выборка: X={X_train.shape}, Y={Y_train_hot.shape}")
    print(f"Тестовая выборка:  X={X_test.shape}, Y={Y_test_hot.shape}")

    # Отрисуем первую попавшуюся цифру из выборки
    sample_idx = 0
    print_ascii_digit(X_train[sample_idx], y_train[sample_idx])
    print(f"One-hot вектор для этой цифры: {Y_train_hot[sample_idx]}")
