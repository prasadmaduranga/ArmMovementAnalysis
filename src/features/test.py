import pandas as pd
import numpy as np

import torch

# create two tensors with batch dimension
batch_size = 32
input_size = 64
encoded_feats=[]
# tensor1 = torch.randn(batch_size, input_size)
# tensor2 = torch.randn(batch_size, input_size)

tensor1 = torch.tensor([[1,2,3],[4,5,6]])
tensor2 =torch.tensor([[10,20,30],[40,50,60]])

# concatenate the tensors along the second dimension
concatenated_tensor = torch.cat([tensor1, tensor2], dim=0)

encoded_feats.extend(torch.cat([tensor1, tensor2], dim=0).tolist())

encoded_feats=np.array(encoded_feats)
np.savetxt('textfile.csv',encoded_feats,delimiter=',')
print(concatenated_tensor)
