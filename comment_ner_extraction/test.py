# import torch
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# if __name__ == '__main__':
#     # for i in range(4,5):
#     #     print(i)
#     # print('*'*100)
#     # for i in range(4,6):
#     #     print(i)
#
#     a = torch.randn(32,65,5)
#     for i in range(32):
#         b = a[i:i+1,5:20,:]
#
#         print(b.shape)
#         c = torch.argmax(b,dim=-1)
#         print(c.shape)
#         print('*'*100)

import time
from transformers import BertModel
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import  torch

if __name__ == '__main__':
    a = torch.randn(2,66,12)
    print(a)
    a.to('cuda')
    b = torch.argmax(a,dim=2)
    print(b.shape)