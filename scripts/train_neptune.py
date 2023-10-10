import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

import neptune

run = neptune.init_run(
    project="mentalakv/wine-recognition",
    api_token=os.getenv('NEPTUNE_API_KEY'),
    capture_stderr=True,
    capture_stdout=True,
    capture_traceback=True,
    capture_hardware_metrics=True
)

params = {
    "lr": 1e-2,
    "bs": 1024,
    "input_sz": 32 * 32 * 3,
    "n_classes": 10,
    "epochs": 50
}
run["parameters"] = params

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
trainset = datasets.CIFAR10("./data", transform=transform, download=True)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=params["bs"], shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on f{device}")


class BaseModel(nn.Module):
    def __init__(self, input_sz, hidden_dim, n_classes):
        super(BaseModel, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_sz, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, input):
        x = input.view(-1, 32 * 32 * 3)
        return self.main(x)


model = BaseModel(params["input_sz"], params["input_sz"], params["n_classes"]).to(device=device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=params["lr"])

for _ in tqdm(range(params["epochs"])):

    model.train()
    for i, (x, y) in enumerate(trainloader, 0):

        x = x.to(device=device)
        y = y.to(device=device)

        optimizer.zero_grad()
        outputs = model.forward(x)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, y)
        acc = (torch.sum(preds == y.data)) / len(x)

        run["train/batch/loss"].append(loss)
        run["train/batch/acc"].append(acc)

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        for X, y in trainloader:
            X = X.to(device=device)
            y = y.to(device=device)

            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        run["valid/acc"] = correct / len(trainloader.dataset)

run.stop()
