import glob
import os
from typing import Optional, Tuple
from typing import Dict, Union, Literal

import imageio
from PIL import Image
import numpy as np
import torch

# from datasets.base.data_proto import CameraInfo, ImageInfo, Rays, ImageMasks
from .base.data_proto import CameraInfo, ImageInfo

import time
import logging
logger = logging.getLogger()


def safe_load_image_tensor(path: str, device: torch.device, is_mask=False, retries=3, sleep_sec=0.5) -> Optional[torch.Tensor]:
    for attempt in range(retries):
        if os.path.exists(path):
            try:
                img = Image.open(path).convert("L" if is_mask else "RGB")
                img_np = np.array(img).astype(np.float32) / 255.0
                if is_mask:
                    img_tensor = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W]
                else:
                    #img_tensor = torch.from_numpy(img_np).permute(2, 1, 0)  # [C, W, H]
                    img_tensor = torch.from_numpy(img_np)
                    #print ("img_tensor shape: ", img_tensor.shape)

                return img_tensor.to(device)
            except Exception as e:
                logger.warning(f"[{attempt+1}/{retries}] Failed to load image at {path}: {e}")
        time.sleep(sleep_sec)
    return None


def safe_load_npy_tensor(path: str, device: torch.device, retries=3, sleep_sec=0.5) -> Optional[torch.Tensor]:
    for attempt in range(retries):
        if os.path.exists(path):
            try:
                data = np.load(path)
                return torch.from_numpy(data).float().to(device)
            except Exception as e:
                logger.warning(f"[{attempt+1}/{retries}] Failed to load .npy at {path}: {e}")
        time.sleep(sleep_sec)
    return None


class NovelViewManager:
    def __init__(self, novel_view_dir: str, debug_mode: bool = True):
        self.novel_view_dir = novel_view_dir
        self.debug_mode = debug_mode

        os.makedirs(self.novel_view_dir, exist_ok=True)

        # if self.debug_mode:
        self.infer_vis_log_dir = os.path.join(self.novel_view_dir, "vis_infer_tmp")
        os.makedirs(self.infer_vis_log_dir, exist_ok=True)

        # whether to use ddp or not
        self.use_ddp = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        self.rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        self.data_store = {}
        self.data_store_name_list = []

    def save_novel_view_data(
        self,
        image_index: int,
        shift_value_name: str,
        novel_view_cam_extrinsic: torch.Tensor,
        novel_view_render_image: torch.Tensor,
        novel_view_render_fix_image: torch.Tensor,
        novel_view_sky_mask: torch.Tensor,
    ):
        """
        Save the novel view data to the disk

        Args:
            image_index: The index of the image
            shift_value_name: The shift value name
            novel_view_cam_extrinsic: The camera extrinsic of the novel view
            novel_view_render_image: The rendered image of the novel view
            novel_view_render_fix_image: The generative image conditioned on the rendered novel view image
            novel_view_sky_mask: The sky mask of the novel view.
        """
        novel_view_render_image = (novel_view_render_image.cpu().numpy() * 255).astype(np.uint8)
        novel_view_render_fix_image = (novel_view_render_fix_image.cpu().numpy() * 255).astype(np.uint8)
        novel_view_sky_mask = (novel_view_sky_mask.cpu().numpy() * 255).astype(np.uint8)

        # Save the image info to the disk
        filename_prefix = f"{image_index:04d}_shift_{shift_value_name}_rank{self.rank}"
        image_path = os.path.join(self.novel_view_dir, f"image_{filename_prefix}.png")
        sky_mask_path = os.path.join(self.novel_view_dir, f"sky_mask_{filename_prefix}.png")
        cam_info_path = os.path.join(self.novel_view_dir, f"cam_extrinsic_{filename_prefix}.npy")
        # import pdb; pdb.set_trace()
        if filename_prefix in self.data_store:
            self.data_store[filename_prefix]['state'] = "ready"
        else:
            print("Error : if filename_prefix in self.data_store:")
            import pdb; pdb.set_trace()

        imageio.imwrite(image_path, novel_view_render_fix_image)
        imageio.imwrite(sky_mask_path, novel_view_sky_mask)

        # Save the camera info to the disk
        np.save(cam_info_path, novel_view_cam_extrinsic.cpu().numpy())

        # Debug mode: save intermediate visualized images to the disk
        # if self.debug_mode:
        vis_img = np.concatenate([novel_view_render_image, novel_view_render_fix_image], axis=1)
        vis_path = os.path.join(self.infer_vis_log_dir, f"gen_vis_{image_index:04d}_shift_{shift_value_name}.png")
        imageio.imwrite(vis_path, vis_img)

        vis_path_degradimg = os.path.join(self.infer_vis_log_dir, f"gen_vis_{image_index:04d}_shift_{shift_value_name}_degradedImg.png")
        imageio.imwrite(vis_path_degradimg, novel_view_render_image)

    def exist_novel_view_data(self, image_index: int, shift_value_name: str):
        """
        Check if the novel view data exists

        Args:
            image_index: The index of the image
            shift_value_name: The pose delta level
        """

        prefix = f"{image_index:04d}_shift_{shift_value_name}_rank{self.rank}"
        image_path = os.path.join(self.novel_view_dir, f"image_{prefix}.png")
        sky_mask_path = os.path.join(self.novel_view_dir, f"sky_mask_{prefix}.png")
        cam_info_path = os.path.join(self.novel_view_dir, f"cam_extrinsic_{prefix}.npy")
        # import pdb; pdb.set_trace()
        # print('exist_novel_view_data::prefix = ', prefix)

        return os.path.exists(image_path) and os.path.exists(sky_mask_path) and os.path.exists(cam_info_path)

    def get_latest_novel_view_shift_value_name(self, image_index: int) -> Optional[str]:
        """
        Get the latest novel view data
        """

        pattern = os.path.join(self.novel_view_dir, f"image_{image_index:04d}_shift_*_rank{self.rank}.png")
        image_files = glob.glob(pattern)
        shift_value_names = [os.path.basename(f).split("_shift_")[1].split(f"_rank{self.rank}")[0] for f in image_files]

        shift_values = [float(name) for name in shift_value_names]
        if len(shift_values) == 0:
            return None
        shift_value = min(shift_values) if shift_values[0] < 0 else max(shift_values)
        shift_value_name = f"{shift_value:.1f}"
        if not self.exist_novel_view_data(image_index, shift_value_name):
            return None
        return shift_value_name


    def load_novel_view_data(
        self,
        image_index: int,
        shift_value_name: str,
        base_image_info: Dict[str, torch.Tensor],
        base_cam_info: Dict[str, torch.Tensor],
        device: torch.device
    ):
        """
        Safe loading of novel view data from disk, with retry logic.
        """
        # shift_value_name = self.get_latest_novel_view_shift_value_name(image_index)
        # if shift_value_name is None:
        #     return None, None

        novel_view_image_info = base_image_info.copy()
        novel_view_cam_info = base_cam_info.copy()

        # File paths
        prefix = f"{image_index:04d}_shift_{shift_value_name}_rank{self.rank}"
        image_path = os.path.join(self.novel_view_dir, f"image_{prefix}.png")
        sky_mask_path = os.path.join(self.novel_view_dir, f"sky_mask_{prefix}.png")
        cam_info_path = os.path.join(self.novel_view_dir, f"cam_extrinsic_{prefix}.npy")

        # Load image
        image_tensor = safe_load_image_tensor(image_path, device=device, is_mask=False)
        if image_tensor is None:
            logger.warning(f"Novel view image not found or unreadable: {image_path}")
            del novel_view_image_info, novel_view_cam_info
            return None, None
        # novel_view_image_info.pixels = image_tensor
        novel_view_image_info["pixels"] = image_tensor

        # Load sky mask
        sky_mask_tensor = safe_load_image_tensor(sky_mask_path, device=device, is_mask=True)
        if sky_mask_tensor is None:
            logger.warning(f"Sky mask not found or unreadable: {sky_mask_path}")
            del novel_view_image_info, novel_view_cam_info
            return None, None
        # novel_view_image_info.masks.sky_mask = sky_mask_tensor.squeeze(0)
        novel_view_image_info["sky_masks"] = sky_mask_tensor.squeeze(0)

        # novel_view_image_info.to(device)

        # Load camera extrinsic
        cam_extrinsic = safe_load_npy_tensor(cam_info_path, device=device)
        if cam_extrinsic is None:
            logger.warning(f"Camera extrinsic not found or unreadable: {cam_info_path}")
            del novel_view_image_info, novel_view_cam_info
            return None, None

        # novel_view_cam_info.camera_to_world = cam_extrinsic
        novel_view_cam_info["camera_to_world"] = cam_extrinsic.to(device)
        # import pdb; pdb.set_trace()
        # novel_view_cam_info.to(device)

        return novel_view_image_info, novel_view_cam_info

    def _get_filename_prefix(self, image_index: int, shift_value_name: str) -> str:
        rank_suffix = f"_rank{self.rank}" if self.use_ddp else ""
        return f"{image_index:04d}_shift_{shift_value_name}{rank_suffix}"

    def push_img_cam_infos(self, image_index: int, shift_value_name: str, state, c2w, nv_img_info = None, nv_cam_info = None) -> None:
        nv_img_name = f"{image_index:04d}_shift_{shift_value_name}_rank{self.rank}"
        previous_count = 0
        # TODOYHH : check 
        if nv_img_name in self.data_store:
            previous_count = self.data_store[nv_img_name]['push_count']
            del self.data_store[nv_img_name]
        self.data_store[nv_img_name] = {
            'state': state,
            'c2w': c2w,
            'push_count': previous_count + 1,
            'nv_img_info': nv_img_info,
            'nv_cam_info': nv_cam_info,
        }
        self.data_store_name_list.append(nv_img_name)

    def update_state(self,  image_index: int, shift_value_name: str, new_state):
        nv_img_name = f"{image_index:04d}_shift_{shift_value_name}_rank{self.rank}"
        if nv_img_name in self.data_store:
            self.data_store[nv_img_name]['state'] = new_state
            return True
        return False 

    def exist_ready_img_cam_infos(self,  image_index: int, shift_value_name: str) -> bool:
        nv_img_name = f"{image_index:04d}_shift_{shift_value_name}_rank{self.rank}"
        if nv_img_name in self.data_store:
            if self.data_store[nv_img_name]['state'] == "ready":
                return True
        return False

    def exist_img_cam_infos(self,  image_index: int, shift_value_name: str) -> bool:
        nv_img_name = f"{image_index:04d}_shift_{shift_value_name}_rank{self.rank}"
        if nv_img_name in self.data_store:
            return True
        return False


    def get_img_cam_infos(self,  image_index: int, shift_value_name: str):
        # import pdb; pdb.set_trace()
        nv_img_name = f"{image_index:04d}_shift_{shift_value_name}_rank{self.rank}"
        if nv_img_name in self.data_store:
            if self.data_store[nv_img_name]['state'] == "ready":
                return self.data_store[nv_img_name]
        return None

    def get_all_names(self):
        return list(self.data_store.keys())