import torch
import torch.nn as nn

batch_size = 2;
time_steps = 3;
embedding_feature = 4;
#这种输入更偏向于NLP领域的数据输入
inputx = torch.randn(batch_size, time_steps, embedding_feature); #N L C

#调用官方API来进行验证
batch_Norm = nn.BatchNorm1d(embedding_feature, affine=False);
batch_test = batch_Norm(inputx.transpose(-1,-2)).transpose(-1,-2);
print(batch_test);

#手写BatchNormalization，代码可以看到求均值与方差是对一整个batch
batch_mean = torch.mean(inputx, dim=(0, 1), keepdim=True);
batch_std = torch.std(inputx, dim=(0, 1), unbiased=False, keepdim=True);
verify_batch = (inputx - batch_mean)/(batch_std+1e-05); #相比官方文档不同，小有误差，可能会影响小数点后五位后
print(verify_batch);