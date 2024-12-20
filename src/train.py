import torch
import torch.nn as nn
import torch.optim as optim
from dvclive import Live
import pandas as pd
import yaml
import os


class LeNet(nn.Module):
    def __init__(self, input_size=4, hidden_size=84, num_classes=3, dropout_rate=0.5):
        super(LeNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def train_model():
    # Загрузка параметров
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["train"]

    # Загрузка данных
    X_train = pd.read_csv("data/X_train.csv").values
    y_train = pd.read_csv("data/y_train.csv").values.ravel()

    # Преобразование в тензоры
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)

    # Инициализация модели
    model = LeNet(
        hidden_size=params["hidden_size"], dropout_rate=params["dropout_rate"]
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    with Live(dir="dvclive_loss", dvcyaml=False) as live:
        for epoch in range(params["epochs"]):
            model.train()
            optimizer.zero_grad()

            outputs = model(X_train)
            loss = criterion(outputs, y_train)

            loss.backward()
            optimizer.step()

            # Логирование метрики
            live.log_metric("loss", loss.item())
            live.next_step()

        # Сохранение модели
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/model.pth")


if __name__ == "__main__":
    train_model()
