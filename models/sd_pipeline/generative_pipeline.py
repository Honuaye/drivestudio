import os

import imageio
import numpy as np
import torch
from einops import rearrange
from .generative_scheduler import GenerativeScheduler
from .sd15_gaussian_fix_model import SD15_GaussianFixModel
from datasets.base.data_proto import CameraInfo, ImageInfo, Rays, ImageMasks

# from ..datasets.base.data_proto import CameraInfo, ImageInfo
# from ..utils.misc import import_str
# from ..utils.training_loop_helper import TrainingLoopHelper

class GenerativeReconTrainer():
    def __init__(self, cfg, devices, num_train_images):
        self.cfg = cfg
        self.device_recon = devices[0]
        self.device_gen = devices[1]
        self.num_train_images = num_train_images
        self.debug_mode = True

        # 1. setup the generative engine
        # GenerativeScheduler : SD 代码调度
        # self.generative_scheduler = import_str(self.cfg.generative_engine.scheduler)()
        self.generative_scheduler = GenerativeScheduler()
        # SD15_GaussianFixModel : SD 修复模型
        pretrained_unet_weights = None
        if "pretrained_unet_weights" in self.cfg.generative_engine:
            pretrained_unet_weights = self.cfg.generative_engine.pretrained_unet_weights
        generative_engine = SD15_GaussianFixModel(
        # generative_engine = import_str(self.cfg.generative_engine.type)(
            base_model_id=self.cfg.generative_engine.base_model_id,
            pretrained_unet_weights=pretrained_unet_weights,
            groundingdino_model_id=self.cfg.generative_engine.groundingdino_model_id,
            sam_model_id=self.cfg.generative_engine.sam_model_id,
            bert_model_id = self.cfg.generative_engine.bert_model_id,
            inference_batch_size=self.cfg.generative_engine.inference_batch_size,
            training_batch_size=self.cfg.generative_engine.training_batch_size,
            use_8bit_optimizer=self.cfg.generative_engine.use_8bit_optimizer,
            dst_size=self.cfg.generative_engine.dst_size,
            num_inference_steps=self.cfg.generative_engine.num_inference_steps,
            image_guidance_scale=self.cfg.generative_engine.image_guidance_scale,
            guidance_scale=self.cfg.generative_engine.guidance_scale,
            multi_gpu=self.cfg.multi_gpu.use_model_parallel,
        )
        # 加载到 self.device_gen 上
        generative_engine.vae.to(self.device_gen)
        generative_engine.text_encoder.to(self.device_gen)
        generative_engine.unet.to(self.device_gen)

        self.generative_scheduler.set_generative_engine(generative_engine)
        self.generative_scheduler.set_train()

        print(f"[CHECK] UNet device: {next(generative_engine.unet.parameters()).device}")
        print(f"[CHECK] VAE device: {next(generative_engine.vae.parameters()).device}")

        # 2. setup joint training logging paths
        # Set generative engine training data vis dirs
        self.train_vis_log_dir = os.path.join(self.cfg.project_dir, "vis_generative_engine_training")
        os.makedirs(self.train_vis_log_dir, exist_ok=True)

        # Set novel view data dirs
        self.novel_view_cache_dir = os.path.join(self.cfg.project_dir, "novel_view_data")
        os.makedirs(self.novel_view_cache_dir, exist_ok=True)

        # 3. setup joint training specific variables
        max_shift = self.cfg.joint_training_cfg.max_shift
        max_num = (
            self.cfg.trainer.optim.num_iters - self.cfg.joint_training_cfg.start_engine_infer_at
        ) // self.cfg.joint_training_cfg.iterations_per_shift
        interval_step = (
            self.cfg.joint_training_cfg.iterations_per_shift
            // num_train_images
            * num_train_images
        )
        self.delta_shift = max_shift / max_num

        start_update_step = self.cfg.joint_training_cfg.start_engine_infer_at
        # import pdb; pdb.set_trace()

        self.update_steps = list(range(start_update_step, self.cfg.trainer.optim.num_iters, interval_step))

        self.current_shift_level = 0
        self.processed_image_indices = set()

    # TODO
    def _forward_interval(self, all_image_info, all_cam_info, from_synthesis=False):
        all_image_info.to(self.device)
        all_cam_info.to(self.device)
        outputs = self.recon_trainer(all_image_info, all_cam_info)
        self.recon_trainer.update_visibility_filter()
        loss_dict = self.recon_trainer.compute_losses(
            outputs=outputs,
            image_info=all_image_info,
            cam_info=all_cam_info,
            from_synthesis=from_synthesis,
        )
        return outputs, loss_dict

    # OK
    def run_before_train_step(self, step):
        # # super().run_before_train_step(step)
        # if self.generative_scheduler.num_novel_data > 0:
        #     self._postprocess_generative_engine_inference_data()
        if self._is_engine_inference_state(step) and step in self.update_steps:
            self.current_shift_level += 1
            self.processed_image_indices.clear()
    # TODO
    def forward_step(self, step, train_data):
        index, image_info, cam_info = train_data

        outputs, loss_dict = self._forward_interval(image_info, cam_info, from_synthesis=False)

        shift_value_name = f"{self.delta_shift * self.current_shift_level:.1f}"
        if self.dataset.exist_novel_view_data(index, shift_value_name):
            novel_view_image_info, novel_view_cam_info = self.dataset.load_novel_view_data(
                index, image_info.detach(), cam_info.detach()
            )
            assert novel_view_image_info is not None and novel_view_cam_info is not None
            novel_view_loss_dict = self._forward_interval(
                novel_view_image_info, novel_view_cam_info, from_synthesis=True
            )[1]
            for k, v in novel_view_loss_dict.items():
                if k not in loss_dict:
                    loss_dict[k] = 0
                loss_dict[k] += v

        # check nan or inf
        for k, v in loss_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in loss {k} at step {step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in loss {k} at step {step}")
        return outputs, loss_dict

    # discard
    def backward_step(self, step, outputs, loss_dict):
        self.recon_trainer.backward(loss_dict)

    # OK
    def run_after_train_step(self, step, train_data, outputs, loss_dict) -> bool:
        # super().run_after_train_step(step, train_data, outputs, loss_dict)
        # import pdb; pdb.set_trace()
        image_info, cam_info = train_data
        # import pdb; pdb.set_trace()
        index = image_info.image_index.item()
        frame_idx = image_info.frame_index[0,0].item()
        camera_id = cam_info.camera_id[0,0].item()
        # print('step = ', step, '; frame_idx = ', frame_idx, '; index = ', index, '; camera_id = ', camera_id)

        if image_info.masks.egocar_mask is not None:
            egocar_mask = 1.0 - image_info.masks.egocar_mask # 为啥 1.0 减法啊
        else:
            egocar_mask = torch.ones(image_info.pixels.shape[:2], device=image_info.pixels.device)

        # 非推理阶段： 训练SD数据，微调。
        if not self._is_engine_inference_state(step):
            self._prepare_generative_engine_training_data(
                step, outputs["rgb"], image_info.pixels, valid_mask=egocar_mask
            )

        # 推理阶段： 准备推理数据； 
        # current_shift_level 的具体含义是什么
        # 问题1：为啥是在 run after train step 里面准备数据呢。 是准备的 下一个 step 数据还是随机的
        if self._is_engine_inference_state(step) and self.current_shift_level > 0:
            # # import pdb; pdb.set_trace()
            # shift = self.delta_shift * self.current_shift_level
            # if self.dataset.exist_novel_view_data(index, f"{shift:.1f}"):
            #     return False
            if index in self.processed_image_indices:
                return False
            return True
            # return self._prepare_generative_engine_inference_data(index, image_info, cam_info, valid_mask=egocar_mask)
        return False

    # OK
    # 判断是否是推理 SD 阶段
    def _is_engine_inference_state(self, step):
        if step == self.cfg.joint_training_cfg.start_engine_infer_at:
            self.generative_scheduler.set_eval()
        return step >= self.cfg.joint_training_cfg.start_engine_infer_at

    # OK
    # 判断是否是训练 SD 阶段
    def _is_engine_training_state(self, step):
        return (
            step >= self.cfg.joint_training_cfg.start_engine_train_at
            and step < self.cfg.joint_training_cfg.stop_engine_train_at
        )

    # OK
    def _prepare_generative_engine_training_data(self, step, render_image, gt_image, valid_mask=None):
        # import pdb; pdb.set_trace()

        # still on device_recon
        render_image = render_image.clamp(0.0, 1.0).detach()
        gt_image = gt_image.clamp(0.0, 1.0).detach()
        valid_mask = valid_mask.detach()

        # record temporary tensors before transfer
        self.tmp_tensors = {
            "render_image": render_image,
            "gt_image": gt_image,
            "valid_mask": valid_mask,
        }

        # move to device_gen
        render_image = (render_image * valid_mask.unsqueeze(-1)).to(self.device_gen)
        gt_image = (gt_image * valid_mask.unsqueeze(-1)).to(self.device_gen)
        valid_mask = valid_mask.to(self.device_gen)

        # delete old tensors on device_recon
        del self.tmp_tensors
        self.clear_device_recon_cache()


        # if self.debug_mode and step % 1000 == 0:
        if self.debug_mode:
            render_img = (render_image.cpu().numpy() * 255).astype(np.uint8)
            gt_img = (gt_image.cpu().numpy() * 255).astype(np.uint8)
            img = np.concatenate([render_img, gt_img], axis=1)
            imageio.imwrite(os.path.join(self.train_vis_log_dir, f"step_{step}_train_pair.png"), img)
            # print(f"Save step_{step}_train_pair.png")

        # all tensors are on device_gen
        render_image = rearrange(render_image, "h w c -> c h w")
        gt_image = rearrange(gt_image, "h w c -> c h w")

        flag = self.generative_scheduler.push_training_pairs(
            render_image=render_image,
            gt_image=gt_image,
            mask=valid_mask,
        )

    # OK
    def _prepare_generative_engine_inference_data(
        self, index, shift, results, c2w, valid_mask=None
    ) -> bool:
        # # import pdb; pdb.set_trace()

        novel_view_render_image = results["rgb"].clamp(0.0, 1.0).detach().clone()
        valid_mask = valid_mask.detach().clone()
        # record temporary tensors to free later
        self.tmp_tensors = {
            "results": results,
            "novel_view_render_image": novel_view_render_image,
            "valid_mask": valid_mask,
            "c2w": c2w,
        }

        # move tensors to device_gen
        novel_view_render_image = (novel_view_render_image * valid_mask.unsqueeze(-1)).to(self.device_gen)
        valid_mask = valid_mask.to(self.device_gen)
        c2w = c2w.to(self.device_gen)

        # free everything from device_recon
        del self.tmp_tensors
        self.clear_device_recon_cache()

        # prepare for generative engine
        novel_view_render_image = rearrange(novel_view_render_image, "h w c -> c h w")
        self.generative_scheduler.push_inference_image(novel_view_render_image, valid_mask, c2w, index, shift)
        self.processed_image_indices.add(index)

    # 这函数单纯是将 修复好的NV图像 从 generative_scheduler 读出来，随后保存到路径内。
        # 问题1 ： 取出来的NV图像是什么时候处理的
        # 问题2 ： 这次step 取出来后保存，同一个 step forward 拿的是相同的图像吗
    # discard
    def _postprocess_generative_engine_inference_data(self):
        import pdb; pdb.set_trace()

        de_img, ref_image, ref_sky_mask, ref_c2w, index = self.generative_scheduler.get_novel_data()

        ref_image = rearrange(ref_image, "c h w -> h w c")
        de_img = rearrange(de_img, "c h w -> h w c")

        shift_value_name = f"{self.delta_shift * self.current_shift_level:.1f}"
        self.dataset.save_novel_view_data(
            index,
            shift_value_name,
            novel_view_cam_extrinsic=ref_c2w,
            novel_view_render_image=de_img,
            novel_view_render_fix_image=ref_image,
            novel_view_sky_mask=ref_sky_mask,
        )

    def clear_device_recon_cache(self):
        import gc
        if self.device_recon.type == "cuda":
            with torch.cuda.device(self.device_recon):
                gc.collect()
                torch.cuda.empty_cache()