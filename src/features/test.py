import torch

# Define a tensor with shape (1, 2, 3, 1)
x = torch.randn(1, 2, 3, 1)

# Remove single-dimensional entries from the shape of x
y = torch.squeeze(x)

# The shape of y will be (2, 3)
print(y.shape)



