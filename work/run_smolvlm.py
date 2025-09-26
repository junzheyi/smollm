import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image

# 自动检测是否有可用的NVIDIA GPU (CUDA)，否则使用CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用 {DEVICE} 设备运行...")
local_model_path = "/data/junzheyi/smollm/work/smolvlm256"

# 1. 加载图片

image1 = Image.open('/data/junzheyi/smollm/work/ny.jpg')
image2 = Image.open('/data/junzheyi/smollm/work/ABUIABACGAAguNSNjAYomKqbjAUwsAk4sAk.jpg')

# 2. 初始化处理器和模型
# 处理器负责将文本和图片转换成模型能理解的格式
print(f"正在从本地路径 '{local_model_path}' 加载模型...")
processor = AutoProcessor.from_pretrained(local_model_path)
# 加载模型本体。torch_dtype=torch.bfloat16 是为了节省显存
# _attn_implementation="flash_attention_2" 使用Flash Attention加速，如果没装或者不用GPU，会自动切换为 "eager"
model = AutoModelForVision2Seq.from_pretrained(
    local_model_path,
    torch_dtype=torch.bfloat16,
    #_attn_implementation="flash_attention_2" if DEVICE == "cuda" else "eager",
    _attn_implementation="sdpa",
    trust_remote_code=True,
).to(DEVICE) # 将模型移动到GPU或CPU
print("模型加载成功！")
# 3. 创建输入信息
# 这是一个类似聊天记录的格式，可以自由组合图片和文字
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"}, # 占位符，对应下面的第一张图片
            {"type": "image"}, # 占位符，对应下面的第二张图片
            {"type": "text", "text": "Can you describe the two images?"} # 你的问题
        ]
    },
]

# 4. 准备模型输入
# 使用处理器的聊天模板来格式化输入
print("正在准备输入数据...")
prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
# 将格式化后的文本和图片列表一起打包成模型输入张量 (tensors)
inputs = processor(text=prompt, images=[image1, image2], return_tensors="pt")
inputs = inputs.to(DEVICE) # 将输入数据也移动到GPU或CPU

# 5. 生成回答
# model.generate 是核心的推理函数，max_new_tokens限制了回答的最大长度
print("正在生成回答...")
generated_ids = model.generate(**inputs, max_new_tokens=500)
# 将模型输出的数字ID解码成人类可读的文本
generated_texts = processor.batch_decode(
    generated_ids,
    skip_special_tokens=True,
)

# 6. 打印结果
print("\n--- 模型输出 ---")
print(generated_texts[0])