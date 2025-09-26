import torch
print(torch.version.cuda)
# 检查 CUDA 是否可用
is_available = torch.cuda.is_available()
print(f"CUDA is available: {is_available}")

if is_available:
    # 打印 PyTorch 检测到的 CUDA 版本
    print(f"PyTorch CUDA version: {torch.version.cuda}")
    # 打印当前 GPU 设备的名称
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")


from transformers import pipeline
classifier = pipeline("sentiment-analysis")
output = classifier("I love using my GPU for deep learning tasks!")
print(output)
