import torch

pthfile = r'checkpoints/handwriting_iter_001.pth' #faster_rcnn_ckpt.pth
net = torch.load(pthfile,map_location=torch.device('cpu'))

print(type(net)) # 类型是 dict
print(len(net)) # 长度为 4，即存在四个 key-value 键值对