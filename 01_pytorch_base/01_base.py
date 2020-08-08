from __future__ import print_function
import torch
import numpy as np


x1 = torch.empty(5, 3)
# print(x1)

x2 = torch.rand(5, 3)
# print(x2)

x3 = torch.zeros(5, 3, dtype=torch.long)
# print(x3)

x4 = torch.tensor([5.5, 3])
# print(x4)

x5 = x4.new_ones(5, 3, dtype=torch.double)
# print(x5)

x6 = torch.randn_like(x5, dtype=torch.float)
# print(x6)

# print(x6.size())

y = torch.rand(5, 3)
# print(x6 + y)
# print(torch.add(x6, y))

result = torch.empty(5, 3)
torch.add(x6, y, out=result)
# print(result)

# add in place
y.add_(x6)
# print(y)

# print(x6[:, 1])

x = torch.rand(4, 4)
y = x.view(16)
z = x.view(-1, 8)
# print(x.size(), y.size(), z.size())

x = torch.rand(1)
# print(x)
# print(x.item())

a = torch.ones(5)
# print(a)

b = a.numpy()
# print(b)

a.add_(1)
# print(a)
# print(b)

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
# print(a)
# print(b)
# CPU上的所有张量(CharTensor除外)都支持与Numpy的相互转换。

if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x, device=device)
    x = x.to("cuda")
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
