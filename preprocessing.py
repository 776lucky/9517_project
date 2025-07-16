import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# === 1. 构造可配对的图像列表 ===
def build_file_lists(rgb_dir, nrg_dir, train_ratio=0.8):
    rgb_files = [f.replace("RGB_", "") for f in os.listdir(rgb_dir) if f.startswith("RGB_")]
    nrg_files = [f.replace("NRG_", "") for f in os.listdir(nrg_dir) if f.startswith("NRG_")]
    common_ids = sorted(set(rgb_files) & set(nrg_files))
    print(f"✅ 找到 {len(common_ids)} 张 RGB 和 NRG 对应图像")

    rgb_filenames = ["RGB_" + fid for fid in common_ids]
    np.random.seed(42)
    np.random.shuffle(rgb_filenames)
    n_train = int(train_ratio * len(rgb_filenames))
    return rgb_filenames[:n_train], rgb_filenames[n_train:]

# === 2. Dataset 定义 ===
class FourChannelSegmentationDataset(Dataset):
    def __init__(self, rgb_dir, nrg_dir, mask_dir, file_list, image_size=(256, 256)):
        self.rgb_dir = rgb_dir
        self.nrg_dir = nrg_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.image_size = image_size

        self.transform_img = T.Compose([
            T.ToPILImage(),
            T.Resize(self.image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        img_id = filename.replace("RGB_", "")
        rgb_path = os.path.join(self.rgb_dir, filename)
        nrg_path = os.path.join(self.nrg_dir, "NRG_" + img_id)
        mask_path = os.path.join(self.mask_dir, "mask_" + img_id)

        rgb = cv2.imread(rgb_path)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        nrg = cv2.imread(nrg_path)
        nir_2d = nrg[:, :, 0]  # 取第一通道作为 NIR

        img_4ch = np.concatenate((rgb, nir_2d[..., None]), axis=-1)

        rgb_tensor = self.transform_img(img_4ch[:, :, :3])
        nir_tensor = self.transform_img(nir_2d)
        img_tensor = torch.cat([rgb_tensor, nir_tensor], dim=0)  # shape: (4, H, W)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.image_size)
        mask = (mask > 127).astype(np.float32)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return img_tensor, mask_tensor


def get_datasets(base_dir="USA_segmentation", image_size=(256, 256)):
    rgb_dir = os.path.join(base_dir, "RGB_images")
    nrg_dir = os.path.join(base_dir, "NRG_images")
    mask_dir = os.path.join(base_dir, "masks")
    train_files, test_files = build_file_lists(rgb_dir, nrg_dir)
    train_dataset = FourChannelSegmentationDataset(rgb_dir, nrg_dir, mask_dir, train_files, image_size)
    test_dataset = FourChannelSegmentationDataset(rgb_dir, nrg_dir, mask_dir, test_files, image_size)
    return train_dataset, test_dataset



# === 3. 使用示例 ===
rgb_dir = "USA_segmentation/RGB_images"
nrg_dir = "USA_segmentation/NRG_images"
mask_dir = "USA_segmentation/masks"

train_files, test_files = build_file_lists(rgb_dir, nrg_dir)

train_dataset = FourChannelSegmentationDataset(rgb_dir, nrg_dir, mask_dir, train_files)
test_dataset = FourChannelSegmentationDataset(rgb_dir, nrg_dir, mask_dir, test_files)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# === 4. 验证是否为 4 通道图像 ===
sample_img, sample_mask = train_dataset[0]
print("✅ 图像 shape:", sample_img.shape)      # 应为 (4, 256, 256)
print("✅ 掩码 shape:", sample_mask.shape)    # 应为 (1, 256, 256)
