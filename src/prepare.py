import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os


def prepare_data():
    # Загрузка данных
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Создание директории для данных
    os.makedirs("data", exist_ok=True)

    # Сохранение данных
    pd.DataFrame(X_train).to_csv("data/X_train.csv", index=False)
    pd.DataFrame(X_test).to_csv("data/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv("data/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv("data/y_test.csv", index=False)


if __name__ == "__main__":
    prepare_data()
