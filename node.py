import torch
import torch.nn as nn
from tqdm import tqdm
from copy import deepcopy
from torch.optim.lr_scheduler import StepLR


class Node:
    def __init__(self, model, dataset) -> None:
        self.model = model
        self.dataset = dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_dataset(self, dataset):
        self.dataset = dataset

    def load_model(self, model):
        self.model = model.to(self.device)

    def load_model_params(self, model_params):
        self.model.load_state_dict(model_params)

    def send_model_params(self):
        return self.model.state_dict()


class Client(Node):
    def __init__(self, id, model=None, dataset=None) -> None:
        super().__init__(model, dataset)
        self.id = id  # client id

    def train(self, epochs):
        batch_size = 64
        learning_rate = 0.001
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = StepLR(
            optimizer, step_size=1, gamma=0.95
        )  # learning rate scheduler, decay rate 0.95
        data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=batch_size, shuffle=True
        )
        with tqdm(range(epochs)) as t:
            t.set_description(f"Client {self.id}")
            for epoch in t:
                t.set_postfix(epoch=epoch)
                for x, y in data_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    optimizer.zero_grad()
                    y_pred = self.model(x)
                    loss = criterion(y_pred, y)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                t.set_postfix(loss=loss.item())


class Server(Node):
    def __init__(
        self,
        clients,
        rounds,
        epochs,
        model=None,
        dataset=None,
    ) -> None:
        super().__init__(model, dataset)
        self.clients = clients
        self.rounds = rounds
        self.epochs = epochs
        if self.model is not None:
            self.keys = self.model.state_dict().keys()

    def load_model(self, model):
        self.model = model.to(self.device)
        self.keys = self.model.state_dict().keys()

    def train(self):
        weights = torch.tensor([len(client.dataset) for client in self.clients])
        weights = weights / weights.sum()
        batch_size = 64
        data_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, batch_size=batch_size, shuffle=False
        )
        for client in self.clients:
            client.load_model(deepcopy(self.model))
        accuracys = []
        for round in range(self.rounds):
            models = []
            model_params = self.model.state_dict()
            for client in self.clients:
                client.load_model_params(model_params)
                client.train(self.epochs)
                models.append(client.send_model_params())
            model_params = self.aggregate(models, weights)
            self.model.load_state_dict(model_params)

            correct = 0
            total = 0
            for x, y in data_loader:
                x = x.to(self.device)
                y_pred = self.model(x)
                _, predicted = torch.max(y_pred.data, 1)
                total += y.size(0)
                correct += (predicted.cpu() == y.cpu()).sum()
            accuracy = 100 * correct.item() / total
            print("Round: {}.  Accuracy: {}".format(round, accuracy))
            accuracys.append(accuracy)
        return accuracys

    def aggregate(self, models, weights):
        new_model_params = {}
        for key in self.keys:
            new_model_params[key] = torch.stack(
                [model[key] * weights[i] for i, model in enumerate(models)]
            ).sum(0)
        return new_model_params
