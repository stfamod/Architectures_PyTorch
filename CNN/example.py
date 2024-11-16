from cnn import CNN_CLAS
from torch import optim, nn
import torch
import matplotlib.pyplot as plt

cnn = CNN_CLAS()
loss_fn = nn.CrossEntropyLoss()
optim = optim.Adam(cnn.parameters(), lr=1e-2)

def train(input, target, iterations):
    losses = []
    for i in range(iterations):
        pred = cnn(input)
        loss = loss_fn(pred, target)
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        losses.append(loss.item())
        if (i % 100 == 0):
            print(f"Loss: {loss.item()}")

    plt.plot(range(iterations), losses)
    plt.title("CNN Classification")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

x = torch.randn((100, 3, 23, 23)) # (100, 3, 23, 23) Batch of 100, 3 Channel, 23x23 tensors.
target = nn.functional.one_hot(torch.randint(0, 9, (1, 100)), num_classes=10).squeeze(0).double() # (100, 10) random one-hot tensors
train(x, target, 1000)