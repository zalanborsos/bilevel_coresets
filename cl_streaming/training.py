import torch
import torch.nn.functional as F


class Training():

    def __init__(self, model, device, nr_epochs, beta=1):
        self.model = model
        self.device = device
        self.nr_epochs = nr_epochs
        self.beta = beta
        self.buffer = []

    def train(self, train_loader):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=5 * 1e-4)
        self.model.train()
        for epoch in range(1, self.nr_epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, target)
                loss.backward()
                optimizer.step()

    def loss(self, output, target):
        loss = F.cross_entropy(output, target)
        for (data, w) in self.buffer:
            X, y = data[0], data[1]
            cs_data = torch.from_numpy(X).to(self.device).type(torch.float)
            cs_target = torch.from_numpy(y).to(self.device).type(torch.long)
            cs_w = torch.from_numpy(w).type(torch.float).to(self.device)
            cs_output = self.model(cs_data)
            loss += self.beta * torch.mean(F.cross_entropy(cs_output, cs_target, reduction='none') * cs_w)
        return loss

    def test(self, test_loader):
        self.model.eval()
        correct = 0
        loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss += F.cross_entropy(output, target, reduction='sum').cpu().item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        return 100. * correct / len(test_loader.dataset)
