import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import tifffile
import glob
import yaml
# data formation
# RI: Nz, Nx, Ny
# value range of RI: delta RI (RI - n_medium)
from skimage.transform import downscale_local_mean
import torch.nn.functional as F



def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def read_tiff(path):
    img_all = []
    count = 0
    print(path)
    paths = list(glob.iglob(path))
    print(paths)
    paths.sort()
    for i in paths:
        print(i)
        img = tifffile.imread(i)
        img_all.append(img.astype(np.float32))
        count+=1
    if(count > 1):
        img_all = np.concatenate(img_all, 0)
    else:
        img_all = img_all[0]
    print(f'image shape{img_all.shape}')
    return torch.from_numpy(img_all)

def clear(img):
    return img.detach().cpu().numpy()

class physics_parameter:
    pass

class ODT_data(Dataset):
    def __init__(self, data_config, physics_config):
        self.physics_param = physics_parameter()
        print(data_config)
        sp_raw, bg_raw = self.read_raw(data_config)
        self.batch_size = data_config['batch_size']
        self.len = sp_raw.shape[0]
        self.sp = sp_raw
        self.bg = bg_raw
        self.wavelength = physics_config['wavelength']
        self.physics_param.wavelength = physics_config['wavelength']
        self.pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
        self.physics_param.pixelsize = physics_config['camera_pixel_size'] / physics_config['magnification']
        self.n_medium = physics_config['n_medium']
        self.physics_param.n_medium = physics_config['n_medium']
        self.NA = physics_config['NA']
        self.physics_param.NA = physics_config['NA']
        self.conjugate_flag = physics_config['conjugate_flag']
        self.angle_num = physics_config['angle_num']
        self.crop_size = data_config['crop_size']
        self.spec_size = data_config['spec_size']
        self.domain_size = data_config['domain_size']
        self.angle_downsample = physics_config['angle_downsample']
        if(physics_config['angle_downsample']):
            num = physics_config['angle_downsample']
            self.sp = self.sp[::num, ...]
            self.bg = self.bg[::num, ...]
            self.angle_num = self.angle_num // num
            print(f'image size downsample to {self.sp.shape}')
        if(self.crop_size[0] < self.sp.shape[1]):
            idx_x = self.sp.shape[1]//2 + data_config['shift_idx'][0]
            idx_y = self.sp.shape[2]//2 + data_config['shift_idx'][1]
            self.sp = self.sp[:, idx_x - self.crop_size[0]//2:idx_x + self.crop_size[0]//2, idx_y - self.crop_size[1]//2:idx_y + self.crop_size[1]//2 ]
            self.bg = self.bg[:, idx_x - self.crop_size[0]//2:idx_x + self.crop_size[0]//2 , idx_y - self.crop_size[1]//2:idx_y + self.crop_size[1]//2 ]
            print(f'image size crop to {self.sp.shape}')

        # # sp = sp[:, ::4, ::4]
        # # 对后两个维度进行4倍降采样
        # # sp = sp[:, ::4, ::4]
        # # 对后两个维度进行4倍降采样
        #
        # # 获取目标长度
        #
        # # 复制填充函数
        # def repeat_frames_to_length(array, target_length):
        #     """
        #     将数组沿第一个维度重复填充到目标长度
        #     每帧重复多次，如：1,1,2,2,3,3...
        #     """
        #     if current_length >= target_length:
        #         return array[:target_length]
        #
        #     # 计算每帧需要重复多少次
        #
        #     # 为每帧创建重复
        #     for i in range(current_length):
        #         # 基础重复次数
        #         # 前remainder帧多重复一次
        #         if i < remainder:
        #
        #         # 重复当前帧
        #         for _ in range(repeats):
        #
        #     return np.stack(repeated_list, axis=0)
        #
        # # 复制填充到相同长度
        #
        # print(
        #     f"After filling - sp_1: {sp_1_filled.shape[0]}, sp_2: {sp_2_filled.shape[0]}, sp_3: {sp_3_filled.shape[0]}")
        #
        # # 准备数据集

        # # 计算全局的颜色范围
        #
        #
        # # 创建GIF的函数
        # def create_sp_gif(datasets, titles, save_path='sp_comparison.gif'):
        #     """
        #     创建显示多个数据集的GIF动画
        #     """
        #
        #     # 初始化图像
        #     for i, (data, title) in enumerate(zip(datasets, titles)):
        #
        #
        #     def animate(frame):
        #         for i, (data, title) in enumerate(zip(datasets, titles)):
        #         return ims
        #
        #     # 创建动画 - 240帧，3秒播放完 = 80 FPS
        #     anim = animation.FuncAnimation(fig, animate, frames=target_length,
        #                                    interval=3000 / target_length, blit=True, repeat=True)
        #
        #     # 保存GIF
        #
        # # 创建单独的GIF
        # def create_individual_gifs(datasets, titles):
        #     """
        #     为每个数据集创建单独的GIF
        #     """
        #     for data, title in zip(datasets, titles):
        #
        #         def animate(frame):
        #
        #         anim = animation.FuncAnimation(fig, animate, frames=target_length,
        #                                        interval=3000 / target_length, blit=False, repeat=True)
        #
        #
        # # 执行GIF创建
        #
        self.exp_name = data_config['expriment_name']
        self.effective_NA = np.minimum(self.NA, self.n_medium)
        self.spec_pixel_size = 2 * np.pi / (self.pixelsize * self.crop_size[0])
        self.k_bound_pixel = np.ceil(self.effective_NA * 2 * np.pi / self.wavelength / self.spec_pixel_size).astype(int)
        self.km_bound_pixel = np.ceil(self.n_medium * 2 * np.pi / self.wavelength / self.spec_pixel_size).astype(int)
        self.km = self.n_medium * 2 * np.pi / self.wavelength
        self.physics_param.km = self.n_medium * 2 * np.pi / self.wavelength
        self.physics_param.k_bound_pixel = np.ceil(self.effective_NA * 2 * np.pi / self.wavelength / self.spec_pixel_size).astype(int)
        self.physics_param.spec_pixel_size = 2 * np.pi / (self.pixelsize * self.crop_size[0])
        print('retrieved pixel size', self.pixelsize * self.crop_size[0] / self.domain_size[-1])


    def read_raw(self, path_config):
        print('reading sample image')
        sp = read_tiff(path_config['sp_path'])
        print('reading background image')
        bg = read_tiff(path_config['bg_path'])
        # return

        return sp, bg
    
    def __len__(self):
            return self.len

    def __getitem__(self, index):
        sp = self.sp[index]
        bg = self.bg[index]
        return sp, bg

    def get_mask_by_var(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        sp_downsample_np = downscale_local_mean(self.sp.cpu().numpy(), (1, 4, 4))
        bg_downsample_np = downscale_local_mean(self.bg.cpu().numpy(), (1, 4, 4))

        sp_downsample = torch.as_tensor(sp_downsample_np, device=device, dtype=torch.float32)
        bg_downsample = torch.as_tensor(bg_downsample_np, device=device, dtype=torch.float32)

        # 1. 沿第 0 维（层方向）计算方差
        # 设置 unbiased=False，使得结果与 numpy 默认行为一致
        sp_var = torch.var(sp_downsample, dim=0, unbiased=False).sqrt()  # shape: [H, W]
        bg_var = torch.var(bg_downsample, dim=0, unbiased=False).sqrt()  # shape: [H, W]

        # 2. 将结果搬到 CPU，便于可视化
        sp_var_cpu = sp_var.detach().cpu().numpy()
        bg_var_cpu = bg_var.detach().cpu().numpy()

        # 3. 可视化两个方差
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        im0 = axes[0].imshow(sp_var_cpu, cmap='viridis')
        axes[0].set_title('Variance of sp_downsample (dim=0)')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.01)

        im1 = axes[1].imshow(bg_var_cpu, cmap='viridis')
        axes[1].set_title('Variance of bg_downsample (dim=0)')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.01)

        plt.tight_layout()
        plt.show()

        # 4. 方差差值（sp_var - bg_var）及其可视化
        var_diff = sp_var - bg_var
        var_diff = var_diff - var_diff.min()
        kernel = torch.ones((1, 1, 3, 3), device=device, dtype=var_diff.dtype) / 9.0
        var_diff_4d = var_diff[None, None, ...]  # 变成 [N=1, C=1, H, W]
        var_diff_pad = F.pad(var_diff_4d, (1, 1, 1, 1), mode='replicate')  # 边缘复制填充
        var_diff_blur = F.conv2d(var_diff_pad, kernel).squeeze(0).squeeze(0)

        # 3) 像素复制式上采样（横纵各 ×2）
        var_diff = var_diff_blur.repeat_interleave(2, dim=0) \
            .repeat_interleave(2, dim=1)
        var_diff_cpu = var_diff.detach().cpu().numpy()

        plt.figure(figsize=(6, 5))
        im2 = plt.imshow(var_diff_cpu, cmap='coolwarm')
        plt.title('Variance difference: sp_var - bg_var')
        plt.axis('off')
        plt.colorbar(im2, fraction=0.046, pad=0.01)
        plt.tight_layout()
        plt.show()

        # 5. 根据差值阈值获取背景掩码
        #    背景通常意味着差值较小；这里给出一个可调阈值示例
        threshold = 7  # 可按需要调整或通过统计方式自动确定
        bg_mask = (torch.abs(var_diff) <= threshold)

        # 6. 可视化掩码
        bg_mask_cpu = bg_mask.detach().cpu().numpy().astype(np.float32)

        plt.figure(figsize=(6, 5))
        plt.imshow(bg_mask_cpu, cmap='gray')
        plt.title(f'Background mask (|var_diff| <= {threshold:g})')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        return bg_mask


def plot_volume(volume, out_path, resolution_pixel):
    v_max = volume.max()
    v_min = volume.min()
    tifffile.imwrite(out_path, volume.detach().cpu().numpy(),
        imagej=True,
        resolution=(1/resolution_pixel[2], 1/resolution_pixel[1]),
        metadata={
            'spacing': resolution_pixel[0],
            'axes': 'ZYX'})
    volume = clear((volume - v_min) / (v_max - v_min))
    print('saving volume slices in ',out_path)
