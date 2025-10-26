#
# Created on Thu Nov 28 2024
#
# Copyright (c) 2024 GigaAI.
#
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.utils.torch_utils import is_compiled_module
from einops import rearrange
from safetensors.torch import load_file
from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer

from .pipeline_gaussian_render_fixer import GaussianRenderFixerPipeline
from .segmentation_pipeline import GroundingDINOPipeline, SAMPipeline

# from ...pipelines.pipeline_gaussian_render_fixer import GaussianRenderFixerPipeline
# from ...utils.model_utils import GroundingDINOPipeline, SAMPipeline

logger = logging.getLogger()


class SD15_GaussianFixModel(nn.Module):
    def __init__(
        self,
        base_model_id: str,
        num_condition: int = 1,
        training_batch_size: int = 4,
        inference_batch_size: int = 4,
        training_learning_rate: float = 5e-5,
        max_norm: Optional[float] = 1.0,
        random_flip_prob: float = 0.5,
        num_inference_steps: int = 20,
        image_guidance_scale: float = -1,
        guidance_scale: float = -1,
        mixed_precision: str = "fp16",
        pretrained_unet_weights: Optional[str] = None,
        groundingdino_model_id: Optional[str] = None,
        sam_model_id: Optional[str] = None,
        bert_model_id: Optional[str] = None,
        use_8bit_optimizer: bool = False,
        dst_size: Optional[Tuple[int, int]] = None,
        multi_gpu: bool = False,
    ):
        super().__init__()
        if mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
        else:
            raise ValueError(f"unsupport mixed_precision {mixed_precision}")

        if multi_gpu:
            self.accelerator = Accelerator(device_placement=False, mixed_precision=mixed_precision)
            self.device = torch.device("cuda:1")
        else:
            self.accelerator = Accelerator(mixed_precision=mixed_precision)
            self.device = self.accelerator.device

        self.base_model_id = base_model_id
        self.noise_scheduler = DDPMScheduler.from_pretrained(base_model_id, subfolder="scheduler",local_files_only = True)
        self.tokenizer = CLIPTokenizer.from_pretrained(base_model_id, subfolder="tokenizer",local_files_only = True)
        self.text_encoder = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder",local_files_only = True)
        self.vae = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae",local_files_only = True)
        self.unet = UNet2DConditionModel.from_pretrained(base_model_id, subfolder="unet",local_files_only = True)
        in_channels = (num_condition + 1) * self.unet.conv_in.in_channels
        self.unet.register_to_config(in_channels=in_channels)
        with torch.no_grad():
            new_conv_in = nn.Conv2d(
                in_channels,
                self.unet.conv_in.out_channels,
                self.unet.conv_in.kernel_size,
                self.unet.conv_in.stride,
                self.unet.conv_in.padding,
            )
            new_conv_in.weight.zero_()
            new_conv_in.weight[:, :4, :, :].copy_(self.unet.conv_in.weight)
            self.unet.conv_in = new_conv_in
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if pretrained_unet_weights is not None:
            self.unet.load_state_dict(load_file(pretrained_unet_weights))
        self.unet.enable_xformers_memory_efficient_attention()
        self.vae.to(self.accelerator.device, dtype=weight_dtype)
        self.text_encoder.to(self.accelerator.device, dtype=weight_dtype)

        # 典型联合使用流程：
        #   image + text → GroundingDINO → bbox → SAM → mask
        # 实现：文本驱动的零样本实例分割
        # 初始化两个模型管道：
            # 1. GroundingDINO 用于文本驱动的边界框检测
            # 2. SAM 用于基于框或点的精确掩码分割
        # 使用文本提示检测物体，输出 bounding box
        self.groundingdino_pipe = GroundingDINOPipeline(
            groundingdino_model_id, bert_model_id, self.accelerator.device, lazy_mode=False
        )
        # 接收图像和 box/point，输出精细的实例分割 mask
        self.sam_pipe = SAMPipeline(sam_model_id, self.accelerator.device, lazy_mode=False)

        if use_8bit_optimizer:
            import bitsandbytes as bnb

            optimizer_cls = bnb.optim.AdamW8bit
        else:
            optimizer_cls = torch.optim.AdamW
        self.optimizer = optimizer_cls(
            self.unet.parameters(),
            lr=training_learning_rate,
        )
        self.unet, self.optimizer = self.accelerator.prepare(self.unet, self.optimizer)
        self.pipe = None

        self.weight_dtype = weight_dtype
        self.max_norm = max_norm
        self.training_batch_size = training_batch_size
        self.inference_batch_size = inference_batch_size
        self.dst_size = dst_size

        self.random_flip_prob = random_flip_prob
        self.num_inference_steps = num_inference_steps
        self.image_guidance_scale = image_guidance_scale
        self.guidance_scale = guidance_scale

        self.prompt_embeds_cache = dict({})

        self.training_forward_times = 0

    def training_forward(self, batch):
        if self.training_forward_times % 100 == 0:
            print("training_forward_times = ", self.training_forward_times)
        loss = self._training_forward_step(*batch)
        self.accelerator.backward(loss)
        if self.max_norm is not None:
            self.accelerator.clip_grad_norm_(self.unet.parameters(), max_norm=self.max_norm)
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.training_forward_times += 1


    # 批量 完成了 NV 视角图像的修复。
    # batch_data ： GS 新视角Render图像
    # masks ： batch_data 图像的天空mask 还是 修复后图像的天空mask ? 初步判断是车前盖 mask
    def inference_forward(self, batch_data, masks, infos, indexes, shifts, prompts, ori_sizes):
        if self.pipe is None:
            self._get_pipe()
        result_list = []
        with torch.no_grad(), torch.autocast(self.accelerator.device.type):
            assert batch_data.is_cuda
            if not (0.0 <= batch_data.mean().item() <= 1.0):
                logger.warning(
                    f"CUDA scheduler warning: batch_data: {batch_data.device}, "
                    f"batch_data.mean(): {batch_data.mean()}, "
                    f"current_indexes: {indexes}. Run cuda sync."
                )
                torch.cuda.synchronize()
            # 这里进行真正的SD修复，novel_view_data 是修复后的图像。
            # batch_data 是一批待处理的GS出来的新视角图像
            novel_view_data = self.pipe(
                prompts,
                image=batch_data,
                num_inference_steps=self.num_inference_steps,
                image_guidance_scale=self.image_guidance_scale,
                guidance_scale=self.guidance_scale,
                output_type="pt",
            ).images
            assert novel_view_data.shape[0] == len(infos)
            for degarded_image, novel_image, mask, info, index, shift, ori_size in zip(
                batch_data, novel_view_data, masks, infos, indexes, shifts, ori_sizes
            ):
                if self.dst_size is not None and ori_size != self.dst_size:
                    novel_image = self._reverse_resize_image(novel_image, ori_size[0], ori_size[1])
                    degarded_image = self._reverse_resize_image(degarded_image, ori_size[0], ori_size[1])

                novel_image = novel_image * mask.unsqueeze(0)

                image_source = (novel_image.cpu().numpy() * 255).astype(np.uint8)
                image_source = rearrange(image_source, "c h w -> h w c")

                boxes = self.groundingdino_pipe(novel_image, caption="sky", box_threshold=0.3, text_threshold=0.25)
                if boxes.shape[0] != 0:
                    _, H, W = novel_image.shape
                    boxes_xyxy = boxes * torch.Tensor([W, H, W, H])
                    boxes_mask = boxes_xyxy[:, 1] < 100  # 100 pixels
                    boxes_xyxy = boxes_xyxy[boxes_mask]
                else:
                    boxes_xyxy = []

                num_boxes = len(boxes_xyxy)
                if num_boxes == 0:
                    sky_mask = np.zeros_like(image_source[..., 0])[None]
                    sky_mask = torch.from_numpy(sky_mask)
                else:
                    masks = self.sam_pipe(image_source, boxes_xyxy)
                    torch.cuda.empty_cache()
                    mask_final = torch.zeros_like(masks[0, 0]).bool()
                    for sky_mask in masks[:, 0]:
                        mask_final = mask_final | sky_mask.bool()
                    sky_mask = mask_final[None]
                sky_mask = sky_mask.squeeze(0).cpu()

                result_list.append((degarded_image, novel_image, sky_mask, info, index, shift))
        return result_list

    def _resize_image(self, img, dst_height, dst_width, mode="bilinear"):
        channel, height, width = img.shape
        scale = min(dst_height / height, dst_width / width)
        new_height, new_width = int(height * scale), int(width * scale)
        resized_img = torch.zeros((channel, dst_height, dst_width), dtype=img.dtype, device=img.device)
        with torch.no_grad():
            resized_original = F.interpolate(img.unsqueeze(0), size=(new_height, new_width), mode=mode).squeeze(0)
        start_h = (dst_height - new_height) // 2
        start_w = (dst_width - new_width) // 2
        resized_img[:, start_h : start_h + new_height, start_w : start_w + new_width] = resized_original
        return resized_img

    def _reverse_resize_image(self, img, ori_height, ori_width, mode="bilinear"):
        _, dst_height, dst_width = img.shape
        scale = min(dst_height / ori_height, dst_width / ori_width)
        new_height, new_width = int(ori_height * scale), int(ori_width * scale)
        start_h = (dst_height - new_height) // 2
        start_w = (dst_width - new_width) // 2
        center_crop = img[:, start_h : start_h + new_height, start_w : start_w + new_width]
        with torch.no_grad():
            original_size_img = F.interpolate(
                center_crop.unsqueeze(0), size=(ori_height, ori_width), mode=mode
            ).squeeze(0)
        return original_size_img

    def get_batch(self, batch_list):
        # Resize images if needed
        for i in range(len(batch_list)):
            if self.dst_size is None or batch_list[i][0].shape[1:] == self.dst_size:
                continue
            dst_height, dst_width = self.dst_size
            resized_data = self._resize_image(batch_list[i][0], dst_height, dst_width)
            resized_gt = self._resize_image(batch_list[i][1], dst_height, dst_width)
            resized_mask = self._resize_image(
                batch_list[i][2].unsqueeze(0), dst_height, dst_width, mode="nearest"
            ).squeeze(0)
            batch_list[i] = (resized_data, resized_gt, resized_mask) + batch_list[i][3:]

        batch_data = 2 * torch.stack([sample[0] for sample in batch_list]) - 1
        batch_gt = 2 * torch.stack([sample[1] for sample in batch_list]) - 1
        batch_mask = torch.stack([sample[2] for sample in batch_list])
        prompts = [sample[3] for sample in batch_list]
        return batch_data, batch_gt, batch_mask, prompts

    def get_infer_batch(self, batch_list):
        ori_sizes = [sample[0].shape[1:] for sample in batch_list]
        # Resize images if needed
        for i in range(len(batch_list)):
            if self.dst_size is None or batch_list[i][0].shape[1:] == self.dst_size:
                continue
            dst_height, dst_width = self.dst_size
            resized_data = self._resize_image(batch_list[i][0], dst_height, dst_width)
            batch_list[i] = (resized_data,) + batch_list[i][1:]
        batch_data = torch.stack([sample[0] for sample in batch_list])
        masks = [sample[1] for sample in batch_list]
        infos = [sample[2] for sample in batch_list]
        normed_time = [sample[3] for sample in batch_list]
        prompts = [sample[4] for sample in batch_list]
        shifts = [sample[5] for sample in batch_list]

        return batch_data, masks, infos, normed_time, shifts, prompts, ori_sizes

    def _prompt_transform(self, prompts: List[str]):
        text_embeds = []
        for prompt in prompts:
            if prompt in self.prompt_embeds_cache:
                text_embeds.append(self.prompt_embeds_cache[prompt])
                continue

            prompt_token = self.tokenizer(
                prompt,
                max_length=self.tokenizer.model_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            text_embeds.append(self.text_encoder(prompt_token)[0])
            self.prompt_embeds_cache[prompt] = text_embeds[-1].clone()
        text_embeds = torch.cat(text_embeds, dim=0)
        return text_embeds

    def _training_forward_step(self, batch_input_data, batch_gt, batch_mask, prompts: List[str]):
        batch_input_data, batch_gt, batch_mask = self._do_data_augmentation(batch_input_data, batch_gt, batch_mask)

        # 转换到指定 device 和精度
        batch_input_data, batch_gt, batch_mask = self.move_batch_to_device(
            batch_input_data, batch_gt, batch_mask, self.device, self.weight_dtype
        )

        image_embeds = self.vae.encode(batch_input_data).latent_dist.mode()
        latents = self.vae.encode(batch_gt).latent_dist.mode()
        latents = latents * self.vae.config.scaling_factor

        noise = torch.randn_like(latents, device=self.device)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (self.training_batch_size,),
            device=self.device,
        ).long()
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)
        conditioned_noisy_lantents = torch.cat(
            [noisy_latents, image_embeds],
            dim=1,
        )

        text_embeds = self._prompt_transform(prompts)

        noise_pred = self.unet(conditioned_noisy_lantents, timesteps, text_embeds, return_dict=False)[0]

        mask_weights = F.max_pool2d(batch_mask.unsqueeze(1), kernel_size=(8, 8))
        noise_pred = noise_pred * mask_weights
        noise = noise * mask_weights
        loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
        return loss

    def _do_data_augmentation(self, batch_input_data, batch_gt, batch_mask):
        # Note: batch_input_data and batch_gt are in GPU memory and the value range is [-1, 1]
        for i in range(batch_input_data.size(0)):
            if torch.rand(1).item() < self.random_flip_prob:
                batch_input_data[i] = torch.flip(batch_input_data[i], dims=[-1])
                batch_gt[i] = torch.flip(batch_gt[i], dims=[-1])
                batch_mask[i] = torch.flip(batch_mask[i], dims=[-1])

        return batch_input_data, batch_gt, batch_mask

    def _get_pipe(self):
        def unwrap_model(model):
            model = self.accelerator.unwrap_model(model)
            model = model._orig_mod if is_compiled_module(model) else model
            return model

        if self.pipe is None:
            self.pipe = GaussianRenderFixerPipeline.from_pretrained(
                self.base_model_id,
                unet=unwrap_model(self.unet),
                text_encoder=unwrap_model(self.text_encoder),
                vae=unwrap_model(self.vae),
                torch_dtype=self.weight_dtype,
            )
            self.pipe.to(self.device)
            self.pipe.set_progress_bar_config(disable=True)

    def set_train(self):
        self.unet.train()
        self.text_encoder.eval()
        self.vae.eval()
        self.pipe = None

    def set_eval(self):
        self.unet.eval()
        self.text_encoder.eval()
        self.vae.eval()
        if self.pipe is None:
            self._get_pipe()

    def move_batch_to_device(self, batch_input_data, batch_gt, batch_mask, device, dtype):
        batch_input_data = batch_input_data.to(device=device, dtype=dtype)
        batch_gt = batch_gt.to(device=device, dtype=dtype)
        batch_mask = batch_mask.to(device=device)
        #torch.cuda.empty_cache()  # optional
        return batch_input_data, batch_gt, batch_mask


