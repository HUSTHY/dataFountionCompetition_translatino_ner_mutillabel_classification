# import torch
# import torch.nn.functional as f
# a = torch.randn(3,5)
# print(a)
# b = torch.where(a>=0.5,1,0)
# print(b)
#
#
# output = 0
# labels = 0
#
# output = torch.sigmoid(output)
# loss = f.binary_cross_entropy(output,labels.float())
#
# # output [B,C]  labels [B,C]
# loss = f.binary_cross_entropy_with_logits(output,labels.float())
#
#
# # output [B,C]  labels [B,C]
# output = torch.softmax(output,dim=1)
# loss = f.binary_cross_entropy(output,labels.float())
#
# # output [B,C]  labels [B,C]
# loss = f.cross_entropy(output,labels)

import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__ == '__main__':
    a = torch.randn(2,3)
    print(a)
    b  = torch.sigmoid(a)
    print(b)
    c = b.to(a)
    print(c)

    d = a.repeat(2,1)
    print(d.shape)
    print(d)


