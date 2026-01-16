# ------------------------------------------------------------
# analysis_fortuna.py
#
# Программа для анализа и прогнозирования показателей работы
# компании ООО «ФОРТУНА». В основе — простая линейная регрессия,
# позволяющая оценить динамику и сделать прогноз на основе
# исторических данных.
#
# Python 3.x
# Используемые библиотеки: pandas, numpy, matplotlib, scikit-learn
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data(path: str) -> pd.DataFrame:
    """
    Загружает данные из CSV-файла.

    :param path: путь к файлу
    :return: DataFrame с исходными данными
    """
    # Читаем CSV как есть, без лишних преобразований
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Минимальная предобработка: убираем строки с пропусками.

    :param df: исходный DataFrame
    :return: очищенный DataFrame
    """
    cleaned = df.dropna()
    return cleaned


def train_model(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """
    Обучает модель линейной регрессии.

    :param X: матрица признаков
    :param y: целевая переменная
    :return: обученная модель
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def evaluate_model(y_true, y_pred):
    """
    Считает базовые метрики качества модели.

    :param y_true: реальные значения
    :param y_pred: предсказанные значения
    :return: (MSE, R^2)
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2


def visualize_results(X, y, y_pred):
    """
    Строит график фактических и прогнозных значений.

    :param X: значения периода
    :param y: реальные данные
    :param y_pred: прогноз модели
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, label="Фактические данные", color="blue")
    plt.plot(X, y_pred, label="Прогноз", color="red", linewidth=2)

    plt.xlabel("Период")
    plt.ylabel("Показатель деятельности")
    plt.title("Анализ и прогноз показателей ООО «ФОРТУНА»")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    """
    Основная логика работы программы:
    загрузка данных → очистка → обучение → оценка → визуализация.
    """

    # Загружаем данные
    data = load_data("data_fortuna.csv")

    # Очищаем от пропусков
    data = preprocess(data)

    # Разделяем признаки и целевую переменную
    X = data[["period"]]
    y = data["value"]

    # Обучаем модель
    model = train_model(X, y)

    # Делаем прогноз
    y_pred = model.predict(X)

    # Оцениваем качество
    mse, r2 = evaluate_model(y, y_pred)

    print("Оценка модели:")
    print(f"MSE: {mse:.2f}")
    print(f"R^2: {r2:.2f}")

    # Показываем график
    visualize_results(X, y, y_pred)


if __name__ == "__main__":
    main()
