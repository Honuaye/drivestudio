import groundingdino.datasets.transforms as T
import numpy as np
import torch
import os
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.inference import predict
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import SamModel, SamProcessor


class LazyPipeline:
    def __init__(self, model_id: str, device: str = None, lazy_mode: bool = True):
        self.model_id = model_id
        self.device = device if device is not None else "cpu"
        self.is_init = False
        self.model = None
        if not lazy_mode:
            self.init_pipeline()

    def init_pipeline(self):
        if not self.is_init:
            self._load_pipeline()
            assert self.model is not None
            if self.device is not None:
                self.model.to(self.device)
            self.is_init = True

    def to(self, device):
        self.device = device
        if self.is_init:
            self.model.to(device)
        return self

    def __call__(self, *args, **kwargs):
        self.init_pipeline()
        return self._forward(*args, **kwargs)

    def _load_pipeline(self):
        raise NotImplementedError

    def _forward(self, *args, **kwargs):
        raise NotImplementedError


class GroundingDINOPipeline(LazyPipeline):
    def __init__(self, groundingdino_model_id: str, bert_model_id:str,device: str = None, lazy_mode: bool = True):
        self.bert_model_id = bert_model_id
        super().__init__(groundingdino_model_id, device, lazy_mode)

    def _load_pipeline(self):
        """
        Load GroundingDINO model
        """

        CKPT_FILENAME = "groundingdino_swinb_cogcoor.pth"
        CKPT_CONFIG_FILENAME = "GroundingDINO_SwinB.cfg.py"
        import os
        # cache_config_file = hf_hub_download(repo_id=self.model_id, filename=CKPT_CONFIG_FILENAME, local_files_only=True)
        cache_config_file = os.path.join(self.model_id, CKPT_CONFIG_FILENAME)
        args = SLConfig.fromfile(cache_config_file)
        args.device = self.device
        args.text_encoder_type = self.bert_model_id # '/pretrained_models/huggingface/models--bert-base-uncased'
        model = build_model(args)

        # cache_file = hf_hub_download(repo_id=self.model_id, filename=CKPT_FILENAME, local_files_only=True)
        cache_file = os.path.join(self.model_id, CKPT_FILENAME)
        checkpoint = torch.load(cache_file, map_location=self.device)
        model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        model.eval()
        self.model = model

    def _forward(
        self,
        image: torch.Tensor,
        caption: str = "sky",
        box_threshold: float = 0.3,
        text_threshold: float = 0.25,
    ) -> torch.Tensor:
        """
        Given a caption, predict boxes from an image using GroundingDINO
        """

        def _normalize_image(image):
            mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(3, 1, 1)
            image_normalized = (image - mean) / std
            return image_normalized

        boxes = predict(
            model=self.model,
            image=_normalize_image(image),
            caption=caption,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )[0]
        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        return boxes


class SAMPipeline(LazyPipeline):
    def __init__(self, sam_model_id: str, device: str = None, lazy_mode: bool = True):
        super().__init__(sam_model_id, device, lazy_mode)

    def _load_pipeline(self):
        """
        Load SAM model
        """
        self.model = SamModel.from_pretrained(self.model_id)
        self.processor = SamProcessor.from_pretrained(self.model_id)

    def _forward(self, image: np.ndarray, boxes: torch.Tensor) -> np.ndarray:
        """
        Given a box, predict masks from an image using SAM
        """
        assert image.ndim == 3
        inputs = self.processor(image, input_boxes=boxes[None], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            masks = self.processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )[0]
        return masks


__all__ = ["GroundingDINOPipeline", "SAMPipeline"]