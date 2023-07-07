import torch
def check_gpu_availability():
    # print cuda and cudnn version
    print(torch.version.cuda)
    print(torch.backends.cudnn.version())
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Number of available GPUs: {device_count}")

        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    else:
        print("No GPUs available.")

if __name__=="__main__":
    check_gpu_availability()