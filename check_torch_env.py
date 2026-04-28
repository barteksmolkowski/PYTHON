import numpy as np
import torch

print(torch.__version__)
matrix = np.zeros((2, 2), dtype=np.int8)
print(matrix)

ai_matrix = torch.tensor(matrix)
print(ai_matrix)