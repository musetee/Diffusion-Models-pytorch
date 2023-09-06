import torch

# print cuda and cudnn version
print('version of torch:',torch.__version__)

print('cuda available:', torch.cuda.is_available())

print('version of cuda:',torch.version.cuda)

print('version of cudnn:',torch.backends.cudnn.version())

# print all GPUs using pytorch

#print(torch.utils.collect_env)

print('device number:',torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f'device {i}:',torch.cuda.get_device_name(i))

print('current device:', torch.cuda.current_device())