# ------------------------------------------------------------
# analysis_fortuna.py
#
# Назначение:
# Программа для анализа и прогнозирования показателей
# деятельности организации ООО «ФОРТУНА» с использованием
# методов машинного обучения.
#
# Язык программирования: Python 3.x
# Используемые библиотеки: pandas, numpy, matplotlib, scikit-learn
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data(path: str) -> pd.DataFrame:
    """
    Функция загрузки исходных данных из CSV-файла.

    :param path: путь к CSV-файлу с данными
    :return: DataFrame с загруженными данными
    """
    data = pd.read_csv(path)
    return data


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Функция предварительной обработки данных.
    Выполняет удаление пропущенных значений.

    :param df: исходный DataFrame
    :return: обработанный DataFrame
    """
    df = df.dropna()
    return df


def train_model(X: pd.DataFrame, y: pd.Series) -> LinearRegression:
    """
    Функция обучения модели линейной регрессии.

    :param X: входные признаки (независимые переменные)
    :param y: целевая переменная
    :return: обученная модель линейной регрессии
    """
    model = LinearRegression()
    model.fit(X, y)
    return model


def evaluate_model(y_true, y_pred):
    """
    Функция оценки качества модели.

    :param y_true: фактические значения показателя
    :param y_pred: прогнозируемые значения
    :return: среднеквадратичная ошибка и коэффициент детерминации
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2


def visualize_results(X, y, y_pred):
    """
    Функция визуализации результатов анализа и прогнозирования.

    :param X: значения временного показателя
    :param y: фактические значения показателя
    :param y_pred: прогнозируемые значения
    """
    plt.figure()
    plt.scatter(X, y, label="Фактические данные")
    plt.plot(X, y_pred, label="Прогноз модели", linewidth=2)
    plt.xlabel("Период")
    plt.ylabel("Показатель деятельности")
    plt.title("Анализ и прогнозирование показателей деятельности ООО «ФОРТУНА»")
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    """
    Главная функция программы.
    Организует последовательное выполнение всех этапов анализа.
    """

    # Загрузка исходных данных
    data = load_data("data_fortuna.csv")

    # Предварительная обработка данных
    data = preprocess(data)

    # Формирование входных и выходных данных
    # period — временной показатель
    # value — анализируемый показатель деятельности
    X = data[["period"]]
    y = data["value"]

    # Обучение модели линейной регрессии
    model = train_model(X, y)

    # Получение прогнозных значений
    y_pred = model.predict(X)

    # Оценка качества модели
    mse, r2 = evaluate_model(y, y_pred)

    # Вывод результатов оценки в консоль
    print("Результаты оценки модели:")
    print(f"Среднеквадратичная ошибка (MSE): {mse:.2f}")
    print(f"Коэффициент детерминации (R^2): {r2:.2f}")

    # Визуализация фактических и прогнозируемых значений
    visualize_results(X, y, y_pred)


# Точка входа в программу
if __name__ == "__main__":
    main()
