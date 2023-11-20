import torch
import scipy
import quant_cuda


batch_size = 32
in_features = 4096
out_features = 4096
dtype = torch.float32


torch.manual_seed(2023)

activations = torch.randn((batch_size, in_features), dtype=dtype, device='cuda')
weights = torch.randn((out_features, in_features), dtype=dtype, device='cuda')


outputs = torch.nn.functional.linear(activations, weights)
torch.cuda.synchronize()

# import ipdb; ipdb.set_trace()
for _ in range(5):
    outputs = torch.nn.functional.linear(activations, weights)
torch.cuda.synchronize()


import time

for _ in range(200):
    outputs = torch.nn.functional.linear(activations, weights)
start = time.time()
for _ in range(1000):
    outputs = torch.nn.functional.linear(activations, weights)
torch.cuda.synchronize()
end = time.time()
print(f'latency: {end - start}')
