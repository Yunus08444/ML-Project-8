"""
ML Project 8 — Variant B
Gradient Boosting with Early Stopping

Что делает скрипт:
  1. Загружает датасет Breast Cancer Wisconsin (Diagnostic) из sklearn.
  2. Делит данные на train / validation / test (60% / 20% / 20%),
     используя ту же схему разделения и random_state=42, что и ML Project 7.
  3. Строит кривые обучения Gradient Boosting (качество на train и validation
     после каждого добавленного дерева).
  4. Реализует Early Stopping по validation set и находит оптимальное число деревьев.
  5. Исследует влияние learning_rate и max_depth (сетка параметров).
  6. Сравнивает Gradient Boosting с Random Forest и Logistic Regression
     на train и test, измеряет время обучения.
  7. Сохраняет таблицы (CSV / Markdown) и графики (PNG) в папку results/.

Важно: test set используется ТОЛЬКО один раз — для финальной оценки.
Подбор гиперпараметров и Early Stopping выполняются исключительно на validation set.
"""

import platform
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # рисуем графики в файлы, без открытия окон

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

# Кириллица в консоли Windows: принудительно выводим в UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# --------------------------------------------------
# 0. Общие настройки
# --------------------------------------------------

RANDOM_STATE = 42          # как в ML Project 7
TEST_SIZE = 0.20           # 20% данных -> test (как в ML Project 7)
VAL_SIZE_OF_REST = 0.25    # 25% от оставшихся 80% = 20% всех данных -> validation

MAX_TREES = 2000           # верхняя граница числа деревьев для Early Stopping
PATIENCE = 30              # сколько деревьев подряд ждать улучшения на validation
TOL = 1e-4                 # минимальное улучшение validation log loss, считающееся улучшением

LEARNING_RATES = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
MAX_DEPTHS = [1, 2, 3, 4, 5, 6, 8]
BASE_LR = 0.1              # learning_rate по умолчанию при исследовании max_depth
BASE_DEPTH = 3             # max_depth по умолчанию при исследовании learning_rate

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def section(title):
    """Печатает заголовок раздела в консоль."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def save_fig(name):
    """Сохраняет текущий график в results/ и закрывает его."""
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / name, dpi=150)
    plt.close()
    print(f"  график сохранён: results/{name}")


# --------------------------------------------------
# 1. Версии библиотек
# --------------------------------------------------

section("ML PROJECT 8 — VARIANT B: GRADIENT BOOSTING WITH EARLY STOPPING")

versions = {
    "python": platform.python_version(),
    "scikit-learn": sklearn.__version__,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "matplotlib": matplotlib.__version__,
}
for lib, ver in versions.items():
    print(f"  {lib:<13}: {ver}")
pd.Series(versions, name="version").to_csv(RESULTS_DIR / "versions.csv", header=True)

# --------------------------------------------------
# 2. Загрузка данных и разделение train / validation / test
# --------------------------------------------------

section("1. ДАННЫЕ")

data = load_breast_cancer(as_frame=True)
X = data.data
y = data.target  # 0 = malignant (злокачественная), 1 = benign (доброкачественная)

print(f"  Размер датасета: {X.shape[0]} объектов, {X.shape[1]} признаков")
print(f"  Пропусков: {int(X.isnull().sum().sum())}")
print(f"  Баланс классов: {dict(y.value_counts().sort_index())} (0=malignant, 1=benign)")

# Сначала отделяем test (20%), затем от оставшихся 80% отделяем validation (25% -> 20% всех).
# stratify=y сохраняет долю классов во всех трёх выборках.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=VAL_SIZE_OF_REST,
    random_state=RANDOM_STATE,
    stratify=y_train_val,
)

print(f"  Train:      {X_train.shape}")
print(f"  Validation: {X_val.shape}")
print(f"  Test:       {X_test.shape}")

# Деревья не требуют масштабирования. Для Logistic Regression масштабирование
# выполняется внутри Pipeline, поэтому StandardScaler обучается только на train
# (утечки данных из validation/test нет).


# --------------------------------------------------
# 3. Вспомогательные функции
# --------------------------------------------------

def staged_curves(model, X_part, y_part):
    """Возвращает log loss и accuracy после каждого дерева (staged-предсказания)."""
    losses, accs = [], []
    for proba in model.staged_predict_proba(X_part):
        losses.append(log_loss(y_part, proba, labels=[0, 1]))
        accs.append(accuracy_score(y_part, proba[:, 1] >= 0.5))
    return np.array(losses), np.array(accs)


def fit_with_early_stopping(learning_rate, max_depth, max_trees=MAX_TREES,
                            patience=PATIENCE, tol=TOL):
    """
    Обучает Gradient Boosting с Early Stopping по validation set.

    Деревья добавляются по одному (warm_start=True). После каждого дерева
    считается log loss на validation. Если в течение `patience` деревьев подряд
    validation log loss не улучшился хотя бы на `tol`, обучение останавливается.

    Возвращает словарь с оптимальным числом деревьев, кривыми и временем.
    Итоговая модель переобучается с n_estimators = оптимальному числу деревьев
    (с тем же random_state она строит в точности те же первые деревья).
    """
    model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=1,
        warm_start=True,
        random_state=RANDOM_STATE,
    )

    val_losses = []
    best_loss = np.inf
    best_n = 0
    rounds_without_improvement = 0

    start = time.perf_counter()
    for n in range(1, max_trees + 1):
        model.n_estimators = n
        model.fit(X_train, y_train)  # с warm_start достраивается только одно новое дерево
        loss = log_loss(y_val, model.predict_proba(X_val), labels=[0, 1])
        val_losses.append(loss)

        if loss < best_loss - tol:
            best_loss = loss
            best_n = n
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
            if rounds_without_improvement >= patience:
                break
    search_time = time.perf_counter() - start

    # Финальная модель с оптимальным числом деревьев.
    final_model = GradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        n_estimators=best_n,
        random_state=RANDOM_STATE,
    )
    start = time.perf_counter()
    final_model.fit(X_train, y_train)
    refit_time = time.perf_counter() - start

    val_proba = final_model.predict_proba(X_val)[:, 1]
    return {
        "learning_rate": learning_rate,
        "max_depth": max_depth,
        "best_n_estimators": best_n,
        "trees_built": len(val_losses),
        "stopped_early": len(val_losses) < max_trees,
        "val_log_loss": log_loss(y_val, val_proba, labels=[0, 1]),
        "val_accuracy": accuracy_score(y_val, val_proba >= 0.5),
        "val_roc_auc": roc_auc_score(y_val, val_proba),
        "early_stopping_time_s": search_time,
        "refit_time_s": refit_time,
        "val_curve": np.array(val_losses),
        "model": final_model,
    }


def evaluate(model, X_part, y_part):
    """Метрики классификации на одной выборке."""
    proba = model.predict_proba(X_part)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_part, pred),
        "f1": f1_score(y_part, pred),
        "roc_auc": roc_auc_score(y_part, proba),
        "log_loss": log_loss(y_part, proba, labels=[0, 1]),
    }


# --------------------------------------------------
# 4. Кривые обучения Gradient Boosting (без остановки)
# --------------------------------------------------

section("2. КРИВЫЕ ОБУЧЕНИЯ: КАЧЕСТВО НА TRAIN И VALIDATION ПО ЧИСЛУ ДЕРЕВЬЕВ")

N_CURVE = 500
curve_model = GradientBoostingClassifier(
    n_estimators=N_CURVE, learning_rate=BASE_LR, max_depth=BASE_DEPTH,
    random_state=RANDOM_STATE,
)
start = time.perf_counter()
curve_model.fit(X_train, y_train)
curve_time = time.perf_counter() - start

train_loss_curve, train_acc_curve = staged_curves(curve_model, X_train, y_train)
val_loss_curve, val_acc_curve = staged_curves(curve_model, X_val, y_val)
best_n_curve = int(np.argmin(val_loss_curve)) + 1

print(f"  Модель: learning_rate={BASE_LR}, max_depth={BASE_DEPTH}, {N_CURVE} деревьев "
      f"(обучение {curve_time:.2f} с)")
print(f"  Минимум validation log loss = {val_loss_curve.min():.4f} при {best_n_curve} деревьях")
print(f"  Validation log loss при {N_CURVE} деревьях = {val_loss_curve[-1]:.4f} "
      f"(рост из-за переобучения)")

trees = np.arange(1, N_CURVE + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].plot(trees, train_loss_curve, label="train")
axes[0].plot(trees, val_loss_curve, label="validation")
axes[0].axvline(best_n_curve, color="gray", ls="--", label=f"минимум val: {best_n_curve} деревьев")
axes[0].set_yscale("log")
axes[0].set_xlabel("Число деревьев")
axes[0].set_ylabel("Log loss (лог. шкала)")
axes[0].set_title("Log loss по числу деревьев")
axes[0].legend()
axes[1].plot(trees, train_acc_curve, label="train")
axes[1].plot(trees, val_acc_curve, label="validation")
axes[1].axvline(best_n_curve, color="gray", ls="--")
axes[1].set_xlabel("Число деревьев")
axes[1].set_ylabel("Accuracy")
axes[1].set_title("Accuracy по числу деревьев")
axes[1].legend()
fig.suptitle(f"Gradient Boosting (learning_rate={BASE_LR}, max_depth={BASE_DEPTH})")
save_fig("01_learning_curves.png")

# --------------------------------------------------
# 5. Early Stopping (базовые параметры)
# --------------------------------------------------

section("3. EARLY STOPPING ПО VALIDATION SET")

base_es = fit_with_early_stopping(BASE_LR, BASE_DEPTH)
print(f"  patience={PATIENCE}, tol={TOL}, максимум {MAX_TREES} деревьев")
print(f"  Построено деревьев до остановки: {base_es['trees_built']}")
print(f"  Оптимальное число деревьев:      {base_es['best_n_estimators']}")
print(f"  Validation log loss: {base_es['val_log_loss']:.4f}, "
      f"accuracy: {base_es['val_accuracy']:.4f}, ROC AUC: {base_es['val_roc_auc']:.4f}")
print(f"  Время Early Stopping: {base_es['early_stopping_time_s']:.2f} с, "
      f"переобучение финальной модели: {base_es['refit_time_s']:.3f} с")

# Для сравнения — встроенный механизм sklearn (n_iter_no_change).
# Он сам отрезает validation_fraction=20% от TRAIN, поэтому test тоже не используется.
builtin = GradientBoostingClassifier(
    n_estimators=MAX_TREES, learning_rate=BASE_LR, max_depth=BASE_DEPTH,
    n_iter_no_change=PATIENCE, validation_fraction=0.2, tol=TOL,
    random_state=RANDOM_STATE,
)
start = time.perf_counter()
builtin.fit(X_train, y_train)
builtin_time = time.perf_counter() - start
print(f"  Встроенный n_iter_no_change: остановился на {builtin.n_estimators_} деревьях "
      f"({builtin_time:.2f} с), validation accuracy = "
      f"{accuracy_score(y_val, builtin.predict(X_val)):.4f}")

es_curve = base_es["val_curve"]
plt.figure(figsize=(8, 4.8))
plt.plot(np.arange(1, len(es_curve) + 1), es_curve, label="validation log loss")
plt.axvline(base_es["best_n_estimators"], color="green", ls="--",
            label=f"лучшее: {base_es['best_n_estimators']} деревьев")
plt.axvline(len(es_curve), color="red", ls=":",
            label=f"остановка: {len(es_curve)} деревьев (patience={PATIENCE})")
plt.xlabel("Число деревьев")
plt.ylabel("Validation log loss")
plt.title("Early Stopping по validation set")
plt.legend()
save_fig("02_early_stopping.png")

pd.DataFrame([
    {"method": "Собственный Early Stopping (validation set)",
     "n_estimators": base_es["best_n_estimators"],
     "trees_built": base_es["trees_built"],
     "val_accuracy": base_es["val_accuracy"],
     "time_s": base_es["early_stopping_time_s"]},
    {"method": "sklearn n_iter_no_change (20% от train)",
     "n_estimators": builtin.n_estimators_,
     "trees_built": builtin.n_estimators_,
     "val_accuracy": accuracy_score(y_val, builtin.predict(X_val)),
     "time_s": builtin_time},
]).round(4).to_csv(RESULTS_DIR / "early_stopping_methods.csv", index=False)

# --------------------------------------------------
# 6. Влияние learning_rate и max_depth (сетка с Early Stopping)
# --------------------------------------------------

section("4. ВЛИЯНИЕ LEARNING_RATE И MAX_DEPTH")

grid_rows = []
grid_curves = {}
total = len(LEARNING_RATES) * len(MAX_DEPTHS)
for i, lr in enumerate(LEARNING_RATES):
    for j, depth in enumerate(MAX_DEPTHS):
        res = fit_with_early_stopping(lr, depth)
        grid_curves[(lr, depth)] = res["val_curve"]
        grid_rows.append({k: v for k, v in res.items() if k not in ("val_curve", "model")})
        print(f"  [{i * len(MAX_DEPTHS) + j + 1:>2}/{total}] lr={lr:<5} depth={depth}: "
              f"best_n={res['best_n_estimators']:>4}, val_log_loss={res['val_log_loss']:.4f}, "
              f"val_acc={res['val_accuracy']:.4f}, time={res['early_stopping_time_s']:.2f} с")

grid_df = pd.DataFrame(grid_rows)
grid_df.round(4).to_csv(RESULTS_DIR / "gb_grid_results.csv", index=False)

# Срезы сетки для отдельных таблиц
lr_df = grid_df[grid_df["max_depth"] == BASE_DEPTH].reset_index(drop=True)
depth_df = grid_df[grid_df["learning_rate"] == BASE_LR].reset_index(drop=True)
lr_df.round(4).to_csv(RESULTS_DIR / "learning_rate_effect.csv", index=False)
depth_df.round(4).to_csv(RESULTS_DIR / "max_depth_effect.csv", index=False)

print("\n  Влияние learning_rate (max_depth=3):")
print(lr_df[["learning_rate", "best_n_estimators", "val_log_loss", "val_accuracy",
             "early_stopping_time_s"]].round(4).to_string(index=False))
print("\n  Влияние max_depth (learning_rate=0.1):")
print(depth_df[["max_depth", "best_n_estimators", "val_log_loss", "val_accuracy",
                "early_stopping_time_s"]].round(4).to_string(index=False))

# График: validation-кривые для разных learning_rate
plt.figure(figsize=(9, 5))
for lr in LEARNING_RATES:
    curve = grid_curves[(lr, BASE_DEPTH)]
    plt.plot(np.arange(1, len(curve) + 1), curve, label=f"lr={lr}")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Число деревьев (лог. шкала)")
plt.ylabel("Validation log loss (лог. шкала)")
plt.title(f"Влияние learning_rate на сходимость (max_depth={BASE_DEPTH})")
plt.legend()
save_fig("03_learning_rate_curves.png")

# График: learning_rate -> оптимальное число деревьев и validation log loss
fig, ax1 = plt.subplots(figsize=(8, 4.8))
ax1.plot(lr_df["learning_rate"], lr_df["best_n_estimators"], "o-", color="tab:blue")
ax1.set_xscale("log")
ax1.set_yscale("log")
ax1.set_xlabel("learning_rate (лог. шкала)")
ax1.set_ylabel("Оптимальное число деревьев", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(lr_df["learning_rate"], lr_df["val_log_loss"], "s--", color="tab:red")
ax2.set_ylabel("Validation log loss", color="tab:red")
plt.title(f"learning_rate: число деревьев и качество (max_depth={BASE_DEPTH})")
save_fig("04_learning_rate_effect.png")

# График: max_depth -> оптимальное число деревьев и validation log loss
fig, ax1 = plt.subplots(figsize=(8, 4.8))
ax1.plot(depth_df["max_depth"], depth_df["best_n_estimators"], "o-", color="tab:blue")
ax1.set_xlabel("max_depth")
ax1.set_ylabel("Оптимальное число деревьев", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(depth_df["max_depth"], depth_df["val_log_loss"], "s--", color="tab:red")
ax2.set_ylabel("Validation log loss", color="tab:red")
plt.title(f"max_depth: число деревьев и качество (learning_rate={BASE_LR})")
save_fig("05_max_depth_effect.png")

# Тепловые карты по всей сетке
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, col, title, fmt in [
    (axes[0], "val_log_loss", "Validation log loss", ".3f"),
    (axes[1], "best_n_estimators", "Оптимальное число деревьев", ".0f"),
]:
    table = grid_df.pivot(index="learning_rate", columns="max_depth", values=col)
    im = ax.imshow(table.values, cmap="viridis_r" if col == "val_log_loss" else "viridis",
                   aspect="auto")
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("learning_rate")
    ax.set_title(title)
    # Цвет подписи выбираем по яркости ячейки, чтобы текст был читаем
    vmin, vmax = np.nanmin(table.values), np.nanmax(table.values)
    for r in range(table.shape[0]):
        for c in range(table.shape[1]):
            value = table.values[r, c]
            share = (value - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            light_cell = share < 0.5 if col == "val_log_loss" else share > 0.5
            ax.text(c, r, format(value, fmt), ha="center", va="center",
                    color="black" if light_cell else "white", fontsize=8)
    fig.colorbar(im, ax=ax)
save_fig("06_grid_heatmaps.png")

# Лучшая комбинация — по минимальному validation log loss (test не участвует!)
best_row = grid_df.loc[grid_df["val_log_loss"].idxmin()]
BEST_LR = float(best_row["learning_rate"])
BEST_DEPTH = int(best_row["max_depth"])
BEST_N = int(best_row["best_n_estimators"])
print(f"\n  Лучшие параметры по validation: learning_rate={BEST_LR}, max_depth={BEST_DEPTH}, "
      f"n_estimators={BEST_N} (val log loss={best_row['val_log_loss']:.4f})")

# --------------------------------------------------
# 7. Подбор Random Forest и Logistic Regression по validation
# --------------------------------------------------

section("5. ПОДБОР RANDOM FOREST И LOGISTIC REGRESSION ПО VALIDATION")

rf_rows = []
for depth in [None, 4, 8, 12]:
    rf = RandomForestClassifier(n_estimators=300, max_depth=depth,
                                random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_rows.append({"max_depth": depth, **{f"val_{k}": v for k, v in evaluate(rf, X_val, y_val).items()}})
rf_df = pd.DataFrame(rf_rows)
best_rf_depth = rf_df.loc[rf_df["val_log_loss"].idxmin(), "max_depth"]
best_rf_depth = None if pd.isna(best_rf_depth) else int(best_rf_depth)
print(rf_df.round(4).to_string(index=False))
print(f"  -> Random Forest: max_depth={best_rf_depth}")

lr_rows = []
for c in [0.01, 0.1, 1.0, 10.0, 100.0]:
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=c, max_iter=5000, random_state=RANDOM_STATE)),
    ])
    logreg.fit(X_train, y_train)
    lr_rows.append({"C": c, **{f"val_{k}": v for k, v in evaluate(logreg, X_val, y_val).items()}})
logreg_df = pd.DataFrame(lr_rows)
best_c = float(logreg_df.loc[logreg_df["val_log_loss"].idxmin(), "C"])
print(logreg_df.round(4).to_string(index=False))
print(f"  -> Logistic Regression: C={best_c}")

rf_df.round(4).to_csv(RESULTS_DIR / "rf_validation.csv", index=False)
logreg_df.round(4).to_csv(RESULTS_DIR / "logreg_validation.csv", index=False)

# --------------------------------------------------
# 8. Финальное сравнение моделей на train и test
# --------------------------------------------------

section("6. СРАВНЕНИЕ МОДЕЛЕЙ НА TRAIN И TEST")

final_models = {
    f"Gradient Boosting + ES (lr={BEST_LR}, depth={BEST_DEPTH}, n={BEST_N})":
        GradientBoostingClassifier(learning_rate=BEST_LR, max_depth=BEST_DEPTH,
                                   n_estimators=BEST_N, random_state=RANDOM_STATE),
    f"Gradient Boosting без ES (lr={BASE_LR}, depth={BASE_DEPTH}, n={N_CURVE})":
        GradientBoostingClassifier(learning_rate=BASE_LR, max_depth=BASE_DEPTH,
                                   n_estimators=N_CURVE, random_state=RANDOM_STATE),
    f"Random Forest (300 деревьев, max_depth={best_rf_depth})":
        RandomForestClassifier(n_estimators=300, max_depth=best_rf_depth,
                               random_state=RANDOM_STATE, n_jobs=-1),
    f"Logistic Regression (C={best_c})":
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(C=best_c, max_iter=5000,
                                             random_state=RANDOM_STATE))]),
}

rows = []
for name, model in final_models.items():
    # Время обучения: среднее по 3 запускам, чтобы уменьшить шум измерения
    times = []
    for _ in range(3):
        start = time.perf_counter()
        model.fit(X_train, y_train)
        times.append(time.perf_counter() - start)
    tr = evaluate(model, X_train, y_train)
    te = evaluate(model, X_test, y_test)
    rows.append({
        "model": name,
        "train_accuracy": tr["accuracy"], "test_accuracy": te["accuracy"],
        "train_f1": tr["f1"], "test_f1": te["f1"],
        "train_roc_auc": tr["roc_auc"], "test_roc_auc": te["roc_auc"],
        "train_log_loss": tr["log_loss"], "test_log_loss": te["log_loss"],
        "accuracy_gap": tr["accuracy"] - te["accuracy"],
        "train_time_s": float(np.mean(times)),
    })

comparison = pd.DataFrame(rows)
comparison.round(4).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
with pd.option_context("display.width", 200, "display.max_columns", 20):
    print(comparison.round(4).to_string(index=False))

# График: accuracy на train и test
short_names = ["GB + ES", "GB без ES", "Random Forest", "Logistic Reg."]
x = np.arange(len(comparison))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
w = 0.38
axes[0].bar(x - w / 2, comparison["train_accuracy"], w, label="train")
axes[0].bar(x + w / 2, comparison["test_accuracy"], w, label="test")
axes[0].set_ylim(0.9, 1.005)
axes[0].set_title("Accuracy: train vs test")
axes[1].bar(x - w / 2, comparison["train_log_loss"], w, label="train")
axes[1].bar(x + w / 2, comparison["test_log_loss"], w, label="test")
axes[1].set_title("Log loss: train vs test (меньше — лучше)")
axes[2].bar(x, comparison["train_time_s"], color="tab:gray")
axes[2].set_title("Время обучения, с")
for ax in axes:
    ax.set_xticks(x, short_names, rotation=15)
axes[0].legend()
axes[1].legend()
save_fig("07_model_comparison.png")

# ROC-кривые и матрицы ошибок на test
fig, axes = plt.subplots(1, len(final_models) + 1, figsize=(22, 4.6))
for (name, model), short in zip(final_models.items(), short_names):
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    axes[0].plot(fpr, tpr, label=f"{short} (AUC={roc_auc_score(y_test, proba):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=0.8)
axes[0].set_xlabel("False positive rate")
axes[0].set_ylabel("True positive rate")
axes[0].set_title("ROC-кривые (test)")
axes[0].legend(fontsize=8)
for ax, (name, model), short in zip(axes[1:], final_models.items(), short_names):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, colorbar=False,
                                          display_labels=["malignant", "benign"])
    ax.set_title(short)
save_fig("08_roc_and_confusion_test.png")

# Важность признаков лучшего Gradient Boosting
gb_best = list(final_models.values())[0]
importance = (pd.Series(gb_best.feature_importances_, index=X.columns)
              .sort_values(ascending=False))
importance.round(4).to_csv(RESULTS_DIR / "gb_feature_importance.csv", header=["importance"])
plt.figure(figsize=(8, 6))
importance.head(15).iloc[::-1].plot.barh()
plt.title("Gradient Boosting + ES: топ-15 признаков")
plt.xlabel("Важность (уменьшение impurity)")
save_fig("09_gb_feature_importance.png")

# --------------------------------------------------
# 9. Итоговый отчёт в Markdown
# --------------------------------------------------

section("7. ИТОГИ")


def md_table(df):
    """Преобразует DataFrame в Markdown-таблицу без внешних зависимостей."""
    df = df.round(4).astype(str)
    lines = ["| " + " | ".join(df.columns) + " |",
             "|" + "|".join(["---"] * len(df.columns)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in df.values]
    return "\n".join(lines)


best_test = comparison.loc[comparison["test_log_loss"].idxmin(), "model"]
summary = f"""# Результаты ML Project 8 — Variant B

Сгенерировано автоматически скриптом `main.py`.

## Версии библиотек
{md_table(pd.DataFrame([versions]))}

## Данные
Breast Cancer Wisconsin (Diagnostic): {X.shape[0]} объектов, {X.shape[1]} признаков.
Train / validation / test = {len(X_train)} / {len(X_val)} / {len(X_test)} (60% / 20% / 20%, stratify, random_state={RANDOM_STATE}).

## Early Stopping (learning_rate={BASE_LR}, max_depth={BASE_DEPTH})
- Построено деревьев до остановки: {base_es['trees_built']} (patience={PATIENCE})
- Оптимальное число деревьев: **{base_es['best_n_estimators']}**
- Validation log loss: {base_es['val_log_loss']:.4f}, accuracy: {base_es['val_accuracy']:.4f}
- Без остановки ({N_CURVE} деревьев) validation log loss = {val_loss_curve[-1]:.4f}, минимум = {val_loss_curve.min():.4f} при {best_n_curve} деревьях

## Влияние learning_rate (max_depth={BASE_DEPTH})
{md_table(lr_df[["learning_rate", "best_n_estimators", "val_log_loss", "val_accuracy", "val_roc_auc", "early_stopping_time_s"]])}

## Влияние max_depth (learning_rate={BASE_LR})
{md_table(depth_df[["max_depth", "best_n_estimators", "val_log_loss", "val_accuracy", "val_roc_auc", "early_stopping_time_s"]])}

## Лучшие параметры Gradient Boosting (по validation)
learning_rate={BEST_LR}, max_depth={BEST_DEPTH}, n_estimators={BEST_N}

## Сравнение моделей на train и test
{md_table(comparison)}

Лучшая модель на test по log loss: **{best_test}**.
"""
(RESULTS_DIR / "summary.md").write_text(summary, encoding="utf-8")
print(f"  Лучшие параметры GB: lr={BEST_LR}, depth={BEST_DEPTH}, n={BEST_N}")
print(f"  Лучшая модель на test по log loss: {best_test}")
print(f"  Все таблицы и графики сохранены в: {RESULTS_DIR}")
