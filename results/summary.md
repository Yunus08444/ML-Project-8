# Результаты ML Project 8 — Variant B

Сгенерировано автоматически скриптом `main.py`.

## Версии библиотек
| python | scikit-learn | numpy | pandas | matplotlib |
|---|---|---|---|---|
| 3.14.3 | 1.8.0 | 2.4.4 | 3.0.2 | 3.10.8 |

## Данные
Breast Cancer Wisconsin (Diagnostic): 569 объектов, 30 признаков.
Train / validation / test = 341 / 114 / 114 (60% / 20% / 20%, stratify, random_state=42).

## Early Stopping (learning_rate=0.1, max_depth=3)
- Построено деревьев до остановки: 77 (patience=30)
- Оптимальное число деревьев: **47**
- Validation log loss: 0.1136, accuracy: 0.9649
- Без остановки (500 деревьев) validation log loss = 0.3143, минимум = 0.1136 при 47 деревьях

## Влияние learning_rate (max_depth=3)
| learning_rate | best_n_estimators | val_log_loss | val_accuracy | val_roc_auc | early_stopping_time_s |
|---|---|---|---|---|---|
| 0.01 | 394 | 0.1239 | 0.9561 | 0.9887 | 1.7803 |
| 0.05 | 100 | 0.1179 | 0.9649 | 0.99 | 0.5096 |
| 0.1 | 47 | 0.1136 | 0.9649 | 0.9902 | 0.2941 |
| 0.2 | 18 | 0.1491 | 0.9561 | 0.9846 | 0.1775 |
| 0.5 | 22 | 0.1686 | 0.9386 | 0.9907 | 0.1971 |
| 1.0 | 8 | 0.1416 | 0.9474 | 0.9905 | 0.1445 |

## Влияние max_depth (learning_rate=0.1)
| max_depth | best_n_estimators | val_log_loss | val_accuracy | val_roc_auc | early_stopping_time_s |
|---|---|---|---|---|---|
| 1 | 387 | 0.074 | 0.9825 | 0.9964 | 1.2601 |
| 2 | 123 | 0.1071 | 0.9649 | 0.9944 | 0.5188 |
| 3 | 47 | 0.1136 | 0.9649 | 0.9902 | 0.2941 |
| 4 | 33 | 0.1502 | 0.9474 | 0.9797 | 0.2654 |
| 5 | 24 | 0.1645 | 0.9561 | 0.9748 | 0.2486 |
| 6 | 24 | 0.1618 | 0.9561 | 0.9743 | 0.2445 |
| 8 | 24 | 0.1651 | 0.9561 | 0.969 | 0.241 |

## Лучшие параметры Gradient Boosting (по validation)
learning_rate=1.0, max_depth=1, n_estimators=38

## Сравнение моделей на train и test
| model | train_accuracy | test_accuracy | train_f1 | test_f1 | train_roc_auc | test_roc_auc | train_log_loss | test_log_loss | accuracy_gap | train_time_s |
|---|---|---|---|---|---|---|---|---|---|---|
| Gradient Boosting + ES (lr=1.0, depth=1, n=38) | 1.0 | 0.9474 | 1.0 | 0.9583 | 1.0 | 0.9911 | 0.0032 | 0.1332 | 0.0526 | 0.0289 |
| Gradient Boosting без ES (lr=0.1, depth=3, n=500) | 1.0 | 0.9386 | 1.0 | 0.9517 | 1.0 | 0.9914 | 0.0 | 0.3527 | 0.0614 | 0.7406 |
| Random Forest (300 деревьев, max_depth=4) | 0.9941 | 0.9474 | 0.9953 | 0.9583 | 0.9996 | 0.9917 | 0.0549 | 0.1233 | 0.0468 | 0.2268 |
| Logistic Regression (C=1.0) | 0.9883 | 0.9825 | 0.9907 | 0.9861 | 0.9972 | 0.995 | 0.0543 | 0.0867 | 0.0058 | 0.004 |

Лучшая модель на test по log loss: **Logistic Regression (C=1.0)**.
