import torch
import pandas as pd
#
#
#
# # create a sample dataframe
# data = {'Company':['GOOG','GOOG','MSFT','MSFT','FB','FB'],
#        'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
#        'Sales':[200,120,340,124,243,350]}
# df = pd.DataFrame(data)
#
# df_grouped = df.groupby('Company')
# print(df_grouped['Company'])
#
#
# # df = pd.read_csv('../Experiments/GCNConv/data/video_graph_features_0.csv')
# # print(df.head())
#
#
# import torch
# from torch_geometric.data import Data
#
#
# x = torch.tensor([[2,1], [5,6], [3,7], [12,0]], dtype=torch.float)
# y = torch.tensor([0, 1, 0, 1], dtype=torch.float)
#
# edge_index = torch.tensor([[0, 2, 1, 0, 3],
#                            [3, 1, 0, 1, 2]], dtype=torch.long)
#
#
# data = Data(x=x, y=y, edge_index=edge_index)
#
import numpy as np
#
# # Initialize a numpy array
# my_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#
# # Convert the numpy array to a list
# my_list = my_array.tolist()

# print(my_list)

x = torch.randn(2,3,2)
print(x)

y = x.reshape(2,6)
print(y)