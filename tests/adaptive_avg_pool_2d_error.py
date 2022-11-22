from torchreid import models, utils
import torch
from torch import nn

# m = nn.AdaptiveAvgPool2d((1, 1))
# input = torch.randn(1, 2048, 16, 8)
# output = m(input)
# print(output.shape)
#
# m = nn.AdaptiveAvgPool2d((1, 1))
# input = torch.randn(36, 2048, 16, 8)
# output = m(input)
# print(output.shape)
#
# m = nn.AdaptiveAvgPool2d((36, 2048, 1, 1))
# input = torch.randn(1, 36, 2048, 16, 8)
# output = m(input)
# print(output.shape)


### TEST AdaptiveAvgPool2d ###
dim = 36
input = torch.randn(4, dim, 2048, 3, 2)

# Solution 1
m1 = nn.AdaptiveAvgPool2d((1, 1))
results = []
for i in range(0, dim):
    slice = input[:, i, :, :, :]
    slice_averaged = m1(slice)
    results.append(torch.unsqueeze(slice_averaged, 1))

output_1 = torch.cat(results, 1)
print(output_1.shape)

# Solution 2
# m2 = nn.AdaptiveAvgPool3d((2048, 1, 1)) # problem, must know 2048 in advance
# output_2 = m2(input)
# print(output_2.shape)

# Solution 3
# m2 = nn.AvgPool3d((1, ?, ?)) # problem, must know kernel size in advance
# output_2 = m2(input)
# print(output_2.shape)

# # Solution 4
# m2 = nn.AdaptiveAvgPool2d((1, 1))
# input_4d = input.view(4*dim, 2048, 3, 2)
# print(input_4d.shape)
# output_2 = m2(input_4d)
# print(output_2.shape)
# output_2 = output_2.view(4, dim, 2048, 1, 1)
# print(output_2.shape)
#
# # Check if valid
# t1 = torch.all(torch.eq(input[0, 0, 0, :, :].flatten().mean(), output_1[0, 0, 0, 0, 0]))
# print(t1)
# t2 = torch.all(torch.eq(input[0, 0, 0, :, :].flatten().mean(), output_2[0, 0, 0, 0, 0]))
# print(t2)
# is_ok = torch.all(torch.eq(output_1, output_2))
# print(is_ok)


#
# # Solution 5
# m2 = nn.AdaptiveAvgPool2d((1, 1))
# # input_4d = input.view(4*dim, 2048, 3, 2)
# input_4d = input.view(input.shape[0]*input.shape[1], input.shape[2], input.shape[3], input.shape[4])
# print(input_4d.shape)
# output_2 = m2(input_4d)
# print(output_2.shape)
# output_2 = output_2.view(input.shape[0], input.shape[1], input.shape[2])
# print(output_2.shape)
#
# output_1 = output_1.squeeze()
# print(output_1.shape)
#
# # Check if valid
# t1 = torch.all(torch.eq(input[0, 0, 0, :, :].flatten().mean(), output_1[0, 0, 0]))
# print(t1)
# t2 = torch.all(torch.eq(input[0, 0, 0, :, :].flatten().mean(), output_2[0, 0, 0]))
# print(t2)
# is_ok = torch.all(torch.eq(output_1, output_2))
# print(is_ok)


t = torch.randn(4, dim, 2048, 3, 2)
unf = t.flatten(0, 1)
print(torch.randn(3, 4, 1).unflatten(1, (2, 2)).shape)
output_2 = unf.unflatten(0, (4, 36))

# Solution 5
m2 = nn.AdaptiveAvgPool2d((1, 1))
# input_4d = input.view(4*dim, 2048, 3, 2)
print(torch.all(torch.eq(input.flatten(0, 1), input.view(-1, input.shape[2], input.shape[3], input.shape[4]))))
# input_4d = input.view(-1, input.shape[2], input.shape[3], input.shape[4])
input_4d = input.flatten(0, 1)
print(input_4d.is_contiguous())
print(input_4d.shape)
output_2 = m2(input_4d)
print(output_2.is_contiguous())
print(output_2.shape)
output_2 = output_2.view(input.shape[0], input.shape[1], -1)
# output_2 = output_2.unflatten(0, (input.shape[0], input.shape[1]))
print(output_2.is_contiguous())
print(output_2.shape)

output_1 = output_1.squeeze()
print(output_1.shape)

# Check if valid
t1 = torch.all(torch.eq(input[0, 0, 0, :, :].flatten().mean(), output_1[0, 0, 0]))
print(t1)
t2 = torch.all(torch.eq(input[0, 0, 0, :, :].flatten().mean(), output_2[0, 0, 0]))
print(t2)
is_ok = torch.all(torch.eq(output_1, output_2))
print(is_ok)