#
# Created on Thu Nov 28 2024
# Author: Wenkang Qin (wkqin@outlook.com)
#
# Copyright (c) 2024 GigaAI.
#
import logging
import threading
from queue import Queue

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