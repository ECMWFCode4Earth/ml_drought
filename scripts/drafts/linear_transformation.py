import torch

#  One linear layer model to map
D_in = 64
model = torch.nn.Sequential(torch.nn.Linear(D_in, 1))

#  loss function
loss_fn = torch.nn.MSELoss(reduction="sum")

# optimizer
learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#  TRAIN
losses = []
for t in range(500):
    y_pred = model(X)
    loss = loss_fn(y_pred, y)

    # train/update the weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.detach().cpu().numpy())
