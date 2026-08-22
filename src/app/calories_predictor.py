# os нужен для проверки файлов и папок.
import os

# Path нужен для удобной работы с путями.
from pathlib import Path

# matplotlib нужен для построения графиков.
import matplotlib.pyplot as plt

# numpy нужен для числовых расчётов.
import numpy as np

# pandas нужен для работы с таблицами.
import pandas as pd

# display красиво показывает таблицы в Google Colab.
from IPython.display import display

# LinearRegression — простая модель машинного обучения для прогноза числа.
from sklearn.linear_model import LinearRegression

# Метрики качества прогноза.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# train_test_split делит данные на обучающую и тестовую части.
from sklearn.model_selection import train_test_split

import seaborn as sns

# Папка для данных.
DATA_DIR = Path("../data")
DATA_DIR.mkdir(exist_ok=True)

# Папка для отчётов.
REPORTS_DIR = Path("../reports")
REPORTS_DIR.mkdir(exist_ok=True)

# Проверяем, что папки созданы.
assert DATA_DIR.exists()
assert REPORTS_DIR.exists()

print("Окружение готово.")

def filter_outliers_iqr_all(df, cols, multiplier=1.5):
    mask = pd.Series([True] * len(df))
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper = min(q3 + multiplier * iqr, 10_000)  # физический предел
        lower = max(q1 - multiplier * iqr, 1)
        mask &= (df[col] >= lower) & (df[col] <= upper)
    return df[mask].reset_index(drop=True)


df = pd.read_parquet(DATA_DIR / "diaries.parquet")
cols_to_clean = [
    "goal_calories",
    "goal_carbs",
    "goal_fat",
    "goal_protein",
    # "goal_sodium",
    "goal_sugar",
    "total_calories",
    "total_carbs",
    "total_fat",
    "total_protein",
    "total_sodium",
    "total_sugar",
]
initial_len = len(df)
df = filter_outliers_iqr_all(df, cols_to_clean, multiplier=1.5)
print(
    f"Удалено строк: {initial_len - len(df)} ({100 * (initial_len - len(df)) / initial_len:.2f}%)"
)


# Размер таблицы.
print("Размер таблицы:", df.shape)

# Названия столбцов.
print("\nНазвания столбцов:")
print(df.columns.tolist())

# Типы данных.
print("\nТипы данных:")
print(df.dtypes)

# Проверка пропусков.
print("\nПропуски по столбцам:")
print(df.isna().sum())

# Описательная статистика.
print("\nОписательная статистика:")
display(df.describe())

# Проверяем, что таблица не пустая.
assert len(df) > 0

# Проверяем, что нет пропусков в ключевых числовых столбцах.
assert (
    df[
        [
            "user_id",
            "date",
            "total_calories",
            "goal_calories",
            "total_carbs",
            "goal_carbs",
            "total_fat",
            "goal_fat",
            "total_protein",
            "goal_protein",
            "total_sodium",
            "goal_sodium",
            "total_sugar",
            "goal_sugar",
            "weekday",
        ]
    ]
    .isna()
    .sum()
    .sum()
    == 0
)


# Целевая переменная
target_column = "total_calories"

# Признаки (все goal-показатели + день недели)
feature_columns = [
    # "goal_calories",
    # "goal_carbs",
    # "goal_fat",
    # "goal_protein",
    # "goal_sodium",
    # "goal_sugar",
    # "weekday",
    "total_carbs",
    "total_fat",
    "total_protein",
    # "total_sugar",
]

# Создаём рабочий DataFrame
model_df = df[feature_columns + [target_column]].copy()

# Показываем первые строки
display(model_df.head())

print("Целевая переменная:", target_column)
print("Признаки:", feature_columns)

# Проверки
assert target_column in model_df.columns
assert set(feature_columns).issubset(model_df.columns)


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sample_size = 50_000
if len(df) > sample_size:
    sample_df = df.sample(n=sample_size, random_state=42)
else:
    sample_df = df

target = "total_calories"
features = ["total_carbs", "total_fat", "total_protein", "total_sugar"]

sns.set_theme(style="whitegrid", palette="muted")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, feat in enumerate(features):
    sns.regplot(
        x=feat,
        y=target,
        data=sample_df,
        ax=axes[i],
        scatter_kws={"alpha": 0.2, "s": 5},
        line_kws={"color": "red", "linewidth": 2},
        ci=None,  # отключаем доверительный интервал для скорости
    )
    axes[i].set_title(f"{target} vs {feat}")
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- 4. Распределение total_calories по дням недели (boxplot + точки) ---
plt.figure(figsize=(10, 6))
sns.boxplot(x="weekday", y=target, data=df, palette="Set3")
sns.stripplot(
    x="weekday",
    y=target,
    data=df.sample(5000, random_state=42),
    color="black",
    alpha=0.2,
    size=2,
)
plt.title("Распределение total_calories по дням недели")
plt.xlabel("День недели (0 = понедельник, 6 = воскресенье)")
plt.ylabel("Калории")
plt.grid(True, alpha=0.3)
plt.show()

# --- 5. Гистограммы с KDE для целевой переменной и одного признака (пример) ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.histplot(df[target], bins=60, kde=True, ax=axes[0], color="blue")
axes[0].set_title(f"Распределение {target}")
axes[0].set_xlabel(target)

sns.histplot(df["total_carbs"], bins=60, kde=True, ax=axes[1], color="green")
axes[1].set_title("Распределение total_carbs")
axes[1].set_xlabel("Углеводы (г)")

plt.tight_layout()
plt.show()

# --- 6. Тепловая карта корреляций между всеми признаками и целью ---
corr_matrix = df[features + [target]].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    square=True,
    cbar_kws={"shrink": 0.8},
)
plt.title("Корреляционная матрица")
plt.show()

# --- 7. Матрица диаграмм рассеяния (pairplot) для всех числовых переменных ---
# Используем только сэмпл, чтобы не перегружать
sns.pairplot(
    sample_df[features + [target]], diag_kind="kde", plot_kws={"alpha": 0.2, "s": 5}
)
plt.suptitle("Парные отношения признаков и целевой переменной", y=1.02)
plt.show()

print("Первичные выводы по графикам:")
print(
    "1. На scatter plots видно, что все макронутриенты имеют сильную положительную линейную связь с калориями."
)
print(
    "2. Распределение калорий по дням недели почти одинаково – день недели, вероятно, не является важным признаком."
)
print(
    "3. Гистограммы показывают, что распределения смещены вправо (много людей потребляют меньше среднего)."
)
print(
    "4. Корреляционная матрица подтверждает высокую корреляцию между углеводами, жирами, белками и калориями."
)
print(
    "5. Сахар коррелирует с калориями слабее остальных – возможно, его влияние уже учтено через углеводы."
)


y = df[target_column]
X = df[feature_columns]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = LinearRegression()

model.fit(X_train, y_train)


print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_)


print("feature_columns =", feature_columns)
model = LinearRegression()
model.fit(X_train, y_train)

print("Свободный член (intercept):", model.intercept_)
print("Коэффициенты модели:")
coef_df = pd.DataFrame({"feature": feature_columns, "coefficient": model.coef_})
display(coef_df)


y_pred_s = model.predict(X_test)

mae_s = mean_absolute_error(y_test, y_pred_s)
rmse_s = np.sqrt(mean_squared_error(y_test, y_pred_s))
r2_s = r2_score(y_test, y_pred_s)

print(f"MAE: {mae_s:.2f} ккал")
print(f"RMSE: {rmse_s:.2f} ккал")
print(f"R²: {r2_s:.3f}")


# Строим прогноз!
y_pred = model.predict(X_test)

# Таблица факт-прогноз.
results_df = pd.DataFrame(
    {
        "real_calories": y_test.values,
        "predicted_calories": y_pred,
    }
)

results_df["error"] = results_df["predicted_calories"] - results_df["real_calories"]
results_df["abs_error"] = results_df["error"].abs()

display(results_df.head(10))

# График факт vs прогноз.
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6)


plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="black",
    linestyle="--",
    linewidth=2,
)

plt.xlabel("Фактический прием пищи ккал")
plt.ylabel("Прогноз модели")
plt.title("Факт vs прогноз")
plt.grid(True)
plt.tight_layout()
plt.show()

# Проверки.
assert len(y_pred) == len(y_test)
assert "abs_error" in results_df.columns


GREEN = "\033[92m"  # ярко-зелёный
RED = "\033[91m"  # ярко-красный
RESET = "\033[0m"  # сброс цвета

# Считаем метрики модели.
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse**0.5
r2 = r2_score(y_test, y_pred)

# Baseline: простой прогноз средним значением обучающей выборки.
baseline_pred = np.full(shape=len(y_test), fill_value=y_train.mean())

# MAE baseline.
baseline_mae = mean_absolute_error(y_test, baseline_pred)

# Печатаем метрики.
print("Метрики модели:")
print("-" * 18)
print("MAE:", round(mae, 3))
print("MSE:", round(mse, 3))
print("RMSE:", round(rmse, 3))
print("R2:", round(r2, 3))
print("-" * 18)
print("Baseline:")
print("Baseline MAE:", round(baseline_mae, 3))
print("-" * 18)

if mae < baseline_mae:
    print(f"{GREEN}Вывод: модель лучше простого прогноза средним.{RESET}")
else:
    print(
        f"{RED}Вывод: модель не лучше простого прогноза средним. Признаки или модель нужно улучшать.{RESET}"
    )


assert mae >= 0
assert mse >= 0
assert rmse >= 0



new_cases = pd.DataFrame(
    {
        "total_carbs": [180, 250, 320],
        "total_fat": [50, 70, 90],
        "total_protein": [70, 100, 130],
    }
)


new_cases["predicted_calories"] = model.predict(new_cases)

display(new_cases)

print("Интерпретация:")
print("Модель прогнозирует фактические калории на основе заданных целей и дня недели.")
print("Чем выше цели по калориям и белкам, тем выше прогноз. День недели влияет слабо.")


import joblib

# Сохраняем модель
joblib.dump(model, "calories_model.joblib")

# Если хотите сохранить и список признаков (чтобы не забыть)
import json

with open("feature_columns.json", "w") as f:
    json.dump(feature_columns, f)


import joblib
import pandas as pd

# Загружаем модель
model = joblib.load("calories_model.joblib")

# Новый объект для прогноза
new_data = pd.DataFrame(
    [
        {
            "total_carbs": 250,
            "total_fat": 70,
            "total_protein": 100,
        }
    ]
)

# Прогноз
prediction = model.predict(new_data[feature_columns])
print(f"Прогноз калорий: {prediction[0]:.0f} ккал")



# Формируем текст отчёта с актуальными данными
# Проверяем, что coef_df не пустой
if len(coef_df) > 0:
    coef_table = coef_df.to_string(index=False)
else:
    coef_table = "Коэффициенты не доступны"

report_text = f"""
# Итоговый отчёт по прогнозированию калорий

## 1. Данные
- Использован датасет дневных записей питания (после очистки выбросов).
- Количество записей: {len(df):,}.
- Признаки: {", ".join(feature_columns)}.
- Целевая переменная: total_calories.

## 2. Задача
Решалась задача регрессии – прогнозирование фактического дневного потребления калорий (`total_calories`) на основе заданных целей по 
макронутриентам.

## 3. Модель
- Использована линейная регрессия (`LinearRegression`).
- Данные разделены на обучающую (80%) и тестовую (20%) выборки.

## 4. Качество модели
- **MAE (средняя абсолютная ошибка):** {mae:.2f} ккал
- **RMSE (среднеквадратичная ошибка):** {rmse:.2f} ккал
- **R² (коэффициент детерминации):** {r2:.3f}
- **Baseline (прогноз средним) MAE:** {baseline_mae:.2f} ккал

Модель {"лучше" if mae < baseline_mae else "не лучше"} простого прогноза средним.

## 5. Коэффициенты модели

**Интерпретация:**
- Положительный коэффициент означает, что увеличение признака ведёт к росту прогнозируемых калорий.
- Отрицательный коэффициент может указывать на обратную связь или мультиколлинеарность.

## 6. Вывод
Модель демонстрирует {"слабую" if r2 < 0.3 else "умеренную" if r2 < 0.6 else "хорошую"} предсказательную способность (R² = {r2:.3f}).
Для улучшения качества рекомендуется добавить индивидуальные признаки пользователей,
временные лаги и использовать регуляризацию или нелинейные модели.

## 7. Применение в дипломном проекте
Структура данного анализа может быть применена к любой задаче регрессии в дипломной работе.
"""

# Сохраняем отчёт
report_path = REPORTS_DIR / "calories_prediction_report.md"
report_path.write_text(report_text, encoding="utf-8")

# Сохраняем таблицу прогнозов (если есть)
predictions_path = REPORTS_DIR / "calories_predictions.csv"
results_df.to_csv(predictions_path, index=False)

print("Отчёт сохранён:", report_path)
print("Таблица прогнозов сохранена:", predictions_path)

print("\n" + "=" * 50)
print(report_text)
print("=" * 50 + "\n")

print("✅ Ноутбук выполнен успешно.")