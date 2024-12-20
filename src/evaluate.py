import torch
import pandas as pd
from train import LeNet
import yaml
from sklearn.metrics import accuracy_score
from dvclive import Live
import matplotlib.pyplot as plt


def evaluate():
    # Загрузка параметров
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    # Загрузка данных
    X_test = pd.read_csv("data/X_test.csv").values
    y_test = pd.read_csv("data/y_test.csv").values.ravel()

    X_test = torch.FloatTensor(X_test)

    # Загрузка модели
    model = LeNet(
        hidden_size=params["hidden_size"], dropout_rate=params["dropout_rate"]
    )
    model.load_state_dict(torch.load("models/model.pth"))
    model.eval()

    # Предсказание
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)

        accuracy = accuracy_score(y_test, predicted.numpy())

    with Live(dir="dvclive_acc", dvcyaml=False) as live:
        live.log_metric("accuracy", accuracy)


if __name__ == "__main__":
    evaluate()
