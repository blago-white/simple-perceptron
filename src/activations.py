import numpy as np


def ReLU(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def ReLU_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(np.float32)


def softmax(z: np.ndarray) -> np.ndarray:
    shifted_z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shifted_z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def compute_loss(y_pred: np.ndarray,
                 y_true_hot: np.ndarray,
                 eps: float = 1e-15) -> float:
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)

    sample_losses = -np.sum(y_true_hot * np.log(y_pred_clipped), axis=-1)
    return float(np.mean(sample_losses))


def test():
    # Быстрый sanity-check функций
    print("--- Тестируем ReLU ---")
    test_z = np.array([-2.0, -0.5, 0.0, 1.5, 3.0])
    print("Вход:       ", test_z)
    print("ReLU:       ", ReLU(test_z))
    print("Производная:", ReLU_derivative(test_z))

    print("\n--- Тестируем Softmax на экстремальных значениях ---")
    large_logits = np.array([[1000.0, 1001.0, 1002.0]])
    probs = softmax(large_logits)
    print("Входные логиты:     ", large_logits)
    print("Выход Softmax:      ", probs)
    print("Сумма вероятностей: ", np.sum(probs))

    print("\n--- Тестируем Loss ---")
    dummy_true = np.array([[0.0, 0.0, 1.0]])
    good_pred = np.array([[0.1, 0.1, 0.8]])
    bad_pred = np.array([[0.8, 0.15, 0.05]])

    print(f"Loss при хорошем ответе: {compute_loss(good_pred, dummy_true):.4f}")
    print(f"Loss при плохом ответе:  {compute_loss(bad_pred, dummy_true):.4f}")