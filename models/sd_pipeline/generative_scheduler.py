#
# Created on Thu Nov 28 2024
# Author: Wenkang Qin (wkqin@outlook.com)
#
# Copyright (c) 2024 GigaAI.
#
import logging
import threading
from queue import Queue
from PIL import Image
import torch

logger = logging.getLogger()


class GenerativeScheduler:
    def __init__(self):
        super().__init__()
        self.training_pairs = []
        self.training_queue = Queue()
        self.inference_images = []
        self.inference_queue = Queue()
        self.novel_data = Queue()
        self.num_infer_samples = 0
        self.num_novel_data = 0

        self.training_stream = torch.cuda.Stream()
        self.inference_stream = torch.cuda.Stream()
        self.data_stream = torch.cuda.Stream()

        self.stop_signal = threading.Event()
        self.train_thread = threading.Thread(target=self._train_worker, daemon=True)
        self.train_thread.start()
        self.infer_thread = threading.Thread(target=self._infer_worker, daemon=True)
        self.infer_thread.start()

    def set_generative_engine(self, generative_engine):
        self.generative_engine = generative_engine
        self.training_batch_size = self.generative_engine.training_batch_size
        self.inference_batch_size = self.generative_engine.inference_batch_size

    def __del__(self):
        if hasattr(self, "stop_signal"):
            self.stop_threads()

    def stop_threads(self):
        if self.stop_signal.is_set():
            return
        self.stop_signal.set()

        # Send the stop signal to queue
        self.training_queue.put(None)
        self.inference_queue.put(None)

        if self.train_thread and self.train_thread.is_alive():
            self.train_thread.join(timeout=5.0)
        if self.infer_thread and self.infer_thread.is_alive():
            self.infer_thread.join(timeout=5.0)

    def _train_worker(self):
        while not self.stop_signal.is_set():
            batch = self.training_queue.get()
            if batch is None:
                break
            with torch.cuda.stream(self.training_stream):
                self.inference_stream.synchronize()
                self.generative_engine.training_forward(batch)

    # 这里完成了 NV 视角图像的修复。
    def _infer_worker(self):
        while not self.stop_signal.is_set():
            batch = self.inference_queue.get()
            if batch is None:
                break
            # TODO
            batch_data, masks, infos, indexes, shifts, prompts, ori_sizes = batch
            self.training_stream.synchronize()
            with torch.cuda.stream(self.inference_stream):
                batch_results = self.generative_engine.inference_forward(
                    batch_data, masks, infos, indexes, shifts, prompts, ori_sizes
                )
                for batch_result in batch_results:
                    # NV视角图像修复完后 存放到 novel data 里面
                    self.novel_data.put(batch_result)
                    self.num_novel_data += 1
                del batch_results

    def push_training_pairs(
        self, render_image, gt_image, mask, prompt: str = "Corrected rendering distortion."
    ) -> bool:
        self.training_pairs.append((render_image.detach(), gt_image.detach(), mask, prompt))

        if self.generative_engine.training and len(self.training_pairs) >= self.training_batch_size:
            batch_list = self.training_pairs[: self.training_batch_size]
            batch_input_data, batch_gt, batch_mask, prompts = self.generative_engine.get_batch(batch_list)
            self.training_pairs = self.training_pairs[self.training_batch_size :]
            self.training_queue.put((batch_input_data, batch_gt, batch_mask, prompts))
            return True
        return False

    # image 这个图像是GS新视角出来的
    # mask 这个mask是自车的车前盖mask，还不是分割的天空mask ！！
    def push_inference_image(self, image, mask, info, index, shift, prompt="Corrected rendering distortion.") -> bool:
        self.inference_images.append((image, mask, info, index, prompt, shift))
        self.num_infer_samples += 1
        # if self.num_infer_samples % self.inference_batch_size == 0:
        #     self.num_novel_data += self.inference_batch_size

        # 触发批量处理
        if len(self.inference_images) >= self.inference_batch_size:
            batch_list = self.inference_images[: self.inference_batch_size]
            # 这里做了什么处理
            batch_data, masks, infos, index, shifts, prompts, ori_sizes = self.generative_engine.get_infer_batch(batch_list)
            assert batch_data.is_cuda
            if not (0.0 <= batch_data.mean().item() <= 1.0):
                logger.warning(
                    f"CUDA scheduler warning: batch_data: {batch_data.device}, "
                    f"batch_data.mean(): {batch_data.mean()}, "
                    f"current_indexes: {index}. Run cuda sync."
                )
                torch.cuda.synchronize()
            # 将批次数据放入推理队列，在 _infer_worker 中会获取这个数据进行 SD 修复。
            self.inference_queue.put((batch_data, masks, infos, index, shifts, prompts, ori_sizes))
            self.inference_images = self.inference_images[self.inference_batch_size :]
            return True
        return False

    def reset_queue(self):
        self.training_pairs.clear()
        self.inference_images.clear()
        while not self.training_queue.empty():
            self.training_queue.get()
        while not self.inference_queue.empty():
            self.inference_queue.get()
        while not self.novel_data:
            self.novel_data.get()
        self.num_infer_samples = 0

    def get_novel_data(self):
        if self.num_novel_data == 0:
            return None
        degrad_data, novel_data, sky_mask, info, index, shift = self.novel_data.get()
        self.num_novel_data -= 1
        return degrad_data, novel_data, sky_mask, info, index, shift

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.stop_threads()

    def set_train(self):
        self.generative_engine.set_train()

    def set_eval(self):
        # Make sure all training data is processed.
        while len(self.training_pairs) > 0:
            self.training_stream.synchronize()
        while not self.training_queue.empty():
            self.training_stream.synchronize()

        self.generative_engine.set_eval()


def load_step_images(folder_path):
    import numpy as np
    from pathlib import Path
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder does not exist or is not a valid directory: {folder_path}")
    image_info = []
    for file in folder.iterdir():
        if not file.is_file():
            continue
        filename = file.name
        if filename.startswith("step_") and filename.endswith("_train_pair.png"):
            try:
                with Image.open(file).convert('RGB') as img:
                    # 转为 (H, W, 3) 格式，与 mask 的 (H, W) 扩展为3通道一致
                    arr = np.array(img, dtype=np.float32)
                    rgb = torch.tensor(arr)
                    # rgb = torch.tensor(arr, device=device)  # 形状 (H, W, 3)
                    # import pdb; pdb.set_trace()
                    image_info.append({"id": int(file.stem.split("_")[1]), "rgb": rgb})
            except (IndexError, ValueError):
                print(f"Warning: Invalid filename format, skipped -> {filename}")
                continue
    image_info.sort(key=lambda x: x["id"])
    return image_info


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, help='Path to the points3D.txt file')
    parser.add_argument('--infer_path', type=str, help='Path to the points3D.txt file')
    args = parser.parse_args()
    train_images_path = args.train_path
    infer_images_path = args.infer_path
    pre_train_unet = False
    if train_images_path is not None:
        train_image_infos = load_step_images(train_images_path)
        pre_train_unet = True
    # import pdb; pdb.set_trace()
    device_recon = torch.device("cuda:0")
    device_gen = torch.device("cuda:1")
    generative_scheduler = GenerativeScheduler()
    base_model_id = "/data/shared/pretrained_models/huggingface/models--runwayml--stable-diffusion-v1-5"
    pretrained_unet_weights = None
    pretrained_unet_weights = "/data/shared/pretrained_models/sd_byd_urban_pretrain/diffusion_pytorch_model.safetensors"
    # pretrained_unet_weights = "/data/workspace/yhh/drivestudio/tmp/trained_unet_refine/nonebase_refine/diffusion_pytorch_model.safetensors"
    # pretrained_unet_weights = "/data/workspace/yhh/drivestudio/tmp/trained_unet_refine/diffusion_pytorch_model.safetensors"
    groundingdino_model_id = "/data/shared/pretrained_models/huggingface/models--ShilongLiu--GroundingDINO"
    sam_model_id = "/data/shared/pretrained_models/huggingface/models--facebook--sam-vit-huge"
    bert_model_id = "/data/shared/pretrained_models/huggingface/models--bert-base-uncased"
    # training_batch_size=1
    training_batch_size=4

    generative_engine = SD15_GaussianFixModel(
        base_model_id=base_model_id,
        pretrained_unet_weights=pretrained_unet_weights,
        groundingdino_model_id=groundingdino_model_id,
        sam_model_id=sam_model_id,
        bert_model_id = bert_model_id,
        # inference_batch_size=4,
        inference_batch_size=1,
        training_batch_size=training_batch_size,
        use_8bit_optimizer=True,
        dst_size=[600, 960],
        num_inference_steps=20,
        image_guidance_scale=-1,
        guidance_scale=-1,
        multi_gpu=True,
        # multi_gpu=self.cfg.multi_gpu.use_model_parallel,
    )

    # 加载到 device_gen 上
    generative_engine.vae.to(device_gen)
    generative_engine.text_encoder.to(device_gen)
    generative_engine.unet.to(device_gen)
    generative_scheduler.set_generative_engine(generative_engine)
    generative_scheduler.set_train()
    # /data/workspace/yhh/drivestudio/output/test_features/
        # data02/refine_waymo_3c_sd_test_10frame_test_new_final_donotDel_test0917_fixBug/vis_generative_engine_training

    if pre_train_unet:
        from einops import rearrange
        from tqdm import tqdm
        for i in tqdm(range(len(train_image_infos)), desc="Processing images"):
            # import pdb; pdb.set_trace()
            rgb = train_image_infos[i]["rgb"]
            if True:
                harf = int(rgb.shape[1] / 2)
                rgb = rgb/255
                rgb_left = rgb[:, :harf, :]  # 形状 [640, 960, 3]
                rgb_right = rgb[:, harf:, :]  # 形状 [640, 960, 3]
                # still on device_recon
                render_image = rgb_left.clamp(0.0, 1.0).detach()
                gt_image = rgb_right.clamp(0.0, 1.0).detach()
                valid_mask = torch.ones(render_image.shape[:2], device=render_image.device)
                valid_mask = valid_mask.detach()
                # record temporary tensors before transfer
                tmp_tensors = {
                    "render_image": render_image,
                    "gt_image": gt_image,
                    "valid_mask": valid_mask,
                }
                # move to device_gen
                render_image = (render_image * valid_mask.unsqueeze(-1)).to(device_gen)
                gt_image = (gt_image * valid_mask.unsqueeze(-1)).to(device_gen)
                valid_mask = valid_mask.to(device_gen)
                # delete old tensors on device_recon
                del tmp_tensors
                # if True:
                #     # self.clear_device_recon_cache()
                #     import gc
                #     if self.device_recon.type == "cuda":
                #         with torch.cuda.device(self.device_recon):
                #             gc.collect()
                #             torch.cuda.empty_cache()
                # all tensors are on device_gen
                render_image = rearrange(render_image, "h w c -> c h w")
                gt_image = rearrange(gt_image, "h w c -> c h w")
            # if i % 500 == 0 :
                # import pdb; pdb.set_trace()

            generative_scheduler.push_training_pairs(
                render_image=render_image,
                gt_image=gt_image,
                mask=valid_mask,
            )
            while (i - (generative_scheduler.generative_engine.training_forward_times * training_batch_size) > 100):
                print("i = ",i,"\t training_forward_times = ",generative_scheduler.generative_engine.training_forward_times)
                print("(i - (generative_scheduler.generative_engine.training_forward_times) < 100)")
                # break

        # import pdb; pdb.set_trace()
        generative_scheduler.set_eval()
        # generative_scheduler.generative_engine.training_forward_times
        # import pdb; pdb.set_trace()
        if True:
            unet_save_path = "./tmp/trained_unet_refine"
            generative_scheduler.generative_engine.unet.save_pretrained(unet_save_path)
        print(f"[CHECK] UNet device: {next(generative_scheduler.generative_engine.unet.parameters()).device}")
        print(f"[CHECK] VAE device: {next(generative_scheduler.generative_engine.vae.parameters()).device}")
    # unet_save_path2 = "./tmp/trained_unet_refine_save2"
    # generative_scheduler.generative_engine.unet.save_pretrained(unet_save_path2)

    # import pdb; pdb.set_trace()
    infer_unet = True
    while(1):
        # from safetensors.torch import load_file
        # pretrained_unet_weights = "/data/shared/pretrained_models/sd_byd_urban_pretrain/diffusion_pytorch_model.safetensors"
        # # pretrained_unet_weights = "/data/workspace/yhh/drivestudio/tmp/trained_unet_refine/diffusion_pytorch_model.safetensors"
        # generative_scheduler.generative_engine.unet.load_state_dict(load_file(pretrained_unet_weights))
        # generative_scheduler.generative_engine.unet.enable_xformers_memory_efficient_attention()
        import pdb; pdb.set_trace()
        if infer_unet:
            import numpy as np
            from einops import rearrange
            generative_scheduler.set_eval()
            index=0
            shift=0.5
            if True:
                degrad_img_path = infer_images_path
                with Image.open(degrad_img_path).convert('RGB') as img:
                    rgb = torch.tensor(np.array(img, dtype=np.float32))
            # import pdb; pdb.set_trace()
            c2w = torch.ones(4, 4)
            novel_view_render_image = (rgb/255).clamp(0.0, 1.0).detach().clone() #todo
            valid_mask = torch.ones(novel_view_render_image.shape[:2], device=novel_view_render_image.device)
            valid_mask = valid_mask.detach().clone()
            # record temporary tensors to free later
            tmp_tensors = {
                "novel_view_render_image": novel_view_render_image,
                "valid_mask": valid_mask,
                "c2w": c2w,
            }
            novel_view_render_image = (novel_view_render_image * valid_mask.unsqueeze(-1)).to(device_gen)
            valid_mask = valid_mask.to(device_gen)
            c2w = c2w.to(device_gen)
            del tmp_tensors
            # self.clear_device_recon_cache()
            novel_view_render_image = rearrange(novel_view_render_image, "h w c -> c h w")
            generative_scheduler.push_inference_image(novel_view_render_image, valid_mask, c2w, index, shift)
            # self.processed_image_indices.add(index)

            import pdb; pdb.set_trace()
            # if True:
            # generative_scheduler.num_novel_data > 0
            while generative_scheduler.num_novel_data > 0:

                de_img, ref_image, ref_sky_mask, ref_c2w, index, shift = generative_scheduler.get_novel_data()
                ref_image = rearrange(ref_image, "c h w -> h w c")
                de_img = rearrange(de_img, "c h w -> h w c")

                de_img = (de_img.cpu().numpy() * 255).astype(np.uint8)
                ref_image = (ref_image.cpu().numpy() * 255).astype(np.uint8)
                import imageio
                import os
                imageio.imwrite(os.path.join(f"tmp/de_img.png"), de_img)
                imageio.imwrite(os.path.join(f"tmp/ref_image.png"), ref_image)
                # img = np.concatenate([render_img, gt_img], axis=1)
                # imageio.imwrite(os.path.join(self.train_vis_log_dir, f"step_{step}_train_pair.png"), img)
                # print(f"Save step_{step}_train_pair.png")

if __name__ == "__main__":
    main()

# Fineturn&&Infer:
    # python -m   models.sd_pipeline.generative_scheduler    --train_path     /data/workspace/yhh/drivestudio/output/test_features/data02/refine_waymo_3c_sd_test_10frame_test_new_final_donotDel_test0917_fixBug/vis_generative_engine_training      --infer_path    /data/workspace/yhh/drivestudio/output/test_features/data02/refine_waymo_3c_sd_test_10frame_closeprint/novel_view_data/vis_infer_tmp/gen_vis_0000_shift_0.4_degradedImg.png

