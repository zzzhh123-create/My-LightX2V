import json

import safetensors.torch

# 加载第一个 block 查看实际键名
block0 = safetensors.torch.load_file("/root/autodl-tmp/LightX2V/models/Wan2.1-I2V-14B-720P-NVFP4/block_0.safetensors")

print("=== block_0.safetensors 中的键名示例 ===")
for i, key in enumerate(list(block0.keys())[:20]):
    print(f"{i}: {key}")

# 查找包含 self_attn.q 的键
print("\n=== 包含 'self_attn.q' 的键 ===")
q_keys = [k for k in block0.keys() if "self_attn.q" in k]
for key in q_keys:
    print(key)

# 检查 input_global_scale 相关键
print("\n=== 包含 'input_global_scale' 的键 ===")
scale_keys = [k for k in block0.keys() if "input_global_scale" in k]
for key in scale_keys:
    print(key)

# 查看 config.json 中的模型配置
with open("/root/autodl-tmp/LightX2V/models/Wan2.1-I2V-14B-720P-NVFP4/config.json", "r") as f:
    config = json.load(f)
print("\n=== config.json 中的模型类型 ===")
print(f"model_type: {config.get('model_type', 'N/A')}")
print(f"_class_name: {config.get('_class_name', 'N/A')}")
