# Four-Channel Semantic Segmentation Preprocessing

---

## ✅ 功能概述

`preprocessing.py` 主要完成以下任务：

1. **图像配对与划分**  
   - 自动匹配 `RGB_` 与 `NRG_` 命名前缀的图像对  
   - 确保每对图像都有对应掩码 `mask_`  
   - 随机划分训练集与测试集（默认 8:2）

2. **四通道图像构建**  
   - RGB 图像读取并转为标准 RGB 格式  
   - 从 NRG 图像中提取第一个通道作为近红外 (NIR)  
   - 拼接为四通道张量 `(R, G, B, NIR)`

3. **图像与掩码标准化**  
   - 所有图像 resize 为统一大小（默认 256×256）  
   - 图像归一化并转为 PyTorch Tensor，shape 为 `(4, H, W)`  
   - 掩码灰度化、二值化，并转为 `(1, H, W)` 格式

4. **Dataset 封装**  
   - 自定义 `FourChannelSegmentationDataset` 类，支持按文件名加载图像对与掩码  
   - 结合 `DataLoader` 支持批量训练流程

---

## 📁 数据目录结构要求

```plaintext
USA_segmentation/
├── RGB_images/
│   └── RGB_XXXX.png
├── NRG_images/
│   └── NRG_XXXX.png
└── masks/
    └── mask_XXXX.png
