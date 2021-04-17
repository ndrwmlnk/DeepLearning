import torch
import torch.nn as nn
import numpy as np

class SimpleNet(nn.Module):

    def __init__(self, input_size, h1, output_size):

        super().__init__()
        self.fc1 = nn.Linear(input_size, h1)
        self.act1 = nn.Sigmoid()  # nn.ReLU()
        self.fc2 = nn.Linear(h1, output_size)
        self.act2 = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.act1(out)
        out = self.fc2(out)
        out = self.act2(out)
        return out

model = SimpleNet(2, 8, 1)
print(model)

# loss = nn.CrossEntropyLoss()
loss = nn.MSELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.1, nesterov=True, momentum=0.9, dampening=0)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []

items = torch.from_numpy(np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype='float32'))

operator = 'xor'
if operator == 'and':
    classes = torch.from_numpy(np.array([[0.0], [0.0], [0.0], [1.0]], dtype='float32'))
elif operator == 'or':
    classes = torch.from_numpy(np.array([[0.0], [1.0], [1.0], [1.0]], dtype='float32'))
elif operator == 'xor':
    classes = torch.from_numpy(np.array([[0.0], [1.0], [1.0], [0.0]], dtype='float32'))

# items = Variable(items)
# classes = Variable(classes)

for epoch in range(10000):

    model.train()  # Put the network into training mode
    optimizer.zero_grad()  # Clear off the gradients from any past operation
    outputs = model(items)  # Do the forward pass
    l = loss(outputs, classes)  # Calculate the loss
    l.backward()  # Calculate the gradients with help of back propagation
    optimizer.step()  # Ask the optimizer to adjust the parameters based on the gradients

    if epoch % 200 == 0:
        print('Epoch %d, Loss: %.4f' % (epoch, l.data.numpy()), [round(o.detach().numpy()[0], 3) for o in outputs])

    model.eval()  # Put the network into evaluation mode
    # Book keeping
    # Record the loss
    train_loss.append(l.data.numpy())

print('\n\n-----------------------')
for name, param in model.named_parameters():
    print(name, param)
print('-----------------------\n\n')
print('DONE')
