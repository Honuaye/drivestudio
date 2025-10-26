from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class Rays:
    origins: torch.Tensor  # [H, W, 3]
    viewdirs: torch.Tensor  # [H, W, 3]
    direction_norm: torch.Tensor  # [H, W, 1]

    def to(self, device: torch.device):
        self.origins = self.origins.to(device, non_blocking=True)
        self.viewdirs = self.viewdirs.to(device, non_blocking=True)
        self.direction_norm = self.direction_norm.to(device, non_blocking=True)
        return self

    def detach(self):
        return Rays(
            origins=self.origins.detach(),
            viewdirs=self.viewdirs.detach(),
            direction_norm=self.direction_norm.detach(),
        )

    def __copy__(self):
        return Rays(
            origins=self.origins,
            viewdirs=self.viewdirs,
            direction_norm=self.direction_norm,
        )

    def copy(self):
        return self.__copy__()


@dataclass
class ImageMasks:
    sky_mask: Optional[torch.Tensor] = None
    dynamic_mask: Optional[torch.Tensor] = None
    human_mask: Optional[torch.Tensor] = None
    vehicle_mask: Optional[torch.Tensor] = None
    egocar_mask: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        if self.sky_mask is not None:
            self.sky_mask = self.sky_mask.to(device, non_blocking=True)
        if self.dynamic_mask is not None:
            self.dynamic_mask = self.dynamic_mask.to(device, non_blocking=True)
        if self.human_mask is not None:
            self.human_mask = self.human_mask.to(device, non_blocking=True)
        if self.vehicle_mask is not None:
            self.vehicle_mask = self.vehicle_mask.to(device, non_blocking=True)
        if self.egocar_mask is not None:
            self.egocar_mask = self.egocar_mask.to(device, non_blocking=True)
        return self

    def detach(self):
        return ImageMasks(
            sky_mask=self.sky_mask.detach() if self.sky_mask is not None else None,
            dynamic_mask=self.dynamic_mask.detach() if self.dynamic_mask is not None else None,
            human_mask=self.human_mask.detach() if self.human_mask is not None else None,
            vehicle_mask=self.vehicle_mask.detach() if self.vehicle_mask is not None else None,
            egocar_mask=self.egocar_mask.detach() if self.egocar_mask is not None else None,
        )

    def __copy__(self):
        return ImageMasks(
            sky_mask=self.sky_mask,
            dynamic_mask=self.dynamic_mask,
            human_mask=self.human_mask,
            vehicle_mask=self.vehicle_mask,
            egocar_mask=self.egocar_mask,
        )

    def copy(self):
        return self.__copy__()


@dataclass
class ImageInfo:
    rays: Rays

    pixel_coords: torch.Tensor  # [H, W, 2]
    image_index: torch.Tensor
    frame_index: torch.Tensor

    pixels: Optional[torch.Tensor] = None  # [H, W, 3]
    masks: Optional[ImageMasks] = None
    depth_map: Optional[torch.Tensor] = None  # [H, W]
    normalized_time: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        self.rays = self.rays.to(device)
        self.pixel_coords = self.pixel_coords.to(device, non_blocking=True)
        self.image_index = self.image_index.to(device)
        self.frame_index = self.frame_index.to(device)

        if self.pixels is not None:
            self.pixels = self.pixels.to(device, non_blocking=True)
        if self.depth_map is not None:
            self.depth_map = self.depth_map.to(device, non_blocking=True)
        if self.masks is not None:
            self.masks = self.masks.to(device)
        if self.normalized_time is not None:
            self.normalized_time = self.normalized_time.to(device, non_blocking=True)

        return self

    def detach(self):
        # rays = Rays(
        #     origins=self.rays.origins.detach(),
        #     viewdirs=self.rays.viewdirs.detach(),
        #     direction_norm=self.rays.direction_norm.detach(),
        # )

        rays = self.rays.detach()
        pixel_coords = self.pixel_coords.detach()
        image_index = self.image_index.detach()
        frame_index = self.frame_index.detach()

        new_info = ImageInfo(rays=rays, pixel_coords=pixel_coords, image_index=image_index, frame_index=frame_index)

        if self.pixels is not None:
            new_info.pixels = self.pixels.detach()

        if self.depth_map is not None:
            new_info.depth_map = self.depth_map.detach()

        if self.normalized_time is not None:
            new_info.normalized_time = self.normalized_time.detach()

        if self.masks is not None:
            masks = ImageMasks()
            if self.masks.sky_mask is not None:
                masks.sky_mask = self.masks.sky_mask.detach()
            if self.masks.dynamic_mask is not None:
                masks.dynamic_mask = self.masks.dynamic_mask.detach()
            if self.masks.human_mask is not None:
                masks.human_mask = self.masks.human_mask.detach()
            if self.masks.vehicle_mask is not None:
                masks.vehicle_mask = self.masks.vehicle_mask.detach()
            if self.masks.egocar_mask is not None:
                masks.egocar_mask = self.masks.egocar_mask.detach()
            new_info.masks = masks

            new_info.masks = self.masks.detach()

        return new_info

    def __copy__(self):
        new_info = ImageInfo(
            rays=self.rays.copy(),
            pixel_coords=self.pixel_coords,
            image_index=self.image_index,
            frame_index=self.frame_index,
        )

        if self.pixels is not None:
            new_info.pixels = self.pixels

        if self.depth_map is not None:
            new_info.depth_map = self.depth_map

        if self.masks is not None:
            new_info.masks = self.masks.copy()

        if self.normalized_time is not None:
            new_info.normalized_time = self.normalized_time

        return new_info

    def copy(self):
        return self.__copy__()


@dataclass
class CameraInfo:
    intrinsic: torch.Tensor  # [3, 3]
    camera_to_world: torch.Tensor  # [4, 4]

    height: int
    width: int

    camera_id: Optional[int] = None
    camera_name: Optional[str] = None
    ego_to_world: Optional[torch.Tensor] = None  # [4, 4]
    camera_to_ego: Optional[torch.Tensor] = None  # [4, 4]

    def to(self, device: torch.device):
        self.intrinsic = self.intrinsic.to(device, non_blocking=True)
        self.camera_to_world = self.camera_to_world.to(device, non_blocking=True)
        if self.ego_to_world is not None:
            self.ego_to_world = self.ego_to_world.to(device, non_blocking=True)
        if self.camera_to_ego is not None:
            self.camera_to_ego = self.camera_to_ego.to(device, non_blocking=True)
        
        return self

    def detach(self):
        intrinsic = self.intrinsic.detach()
        camera_to_world = self.camera_to_world.detach()
        new_info = CameraInfo(
            intrinsic=intrinsic,
            camera_to_world=camera_to_world,
            height=self.height,
            width=self.width,
            camera_id=self.camera_id,
            camera_name=self.camera_name,
        )
        if self.ego_to_world is not None:
            new_info.ego_to_world = self.ego_to_world.detach()
        if self.camera_to_ego is not None:
            new_info.camera_to_ego = self.camera_to_ego.detach()
        return new_info

    def __copy__(self):
        new_info = CameraInfo(
            intrinsic=self.intrinsic,
            camera_to_world=self.camera_to_world,
            height=self.height,
            width=self.width,
            camera_id=self.camera_id,
            camera_name=self.camera_name,
        )

        if self.ego_to_world is not None:
            new_info.ego_to_world = self.ego_to_world

        if self.camera_to_ego is not None:
            new_info.camera_to_ego = self.camera_to_ego

        return new_info

    def copy(self):
        return self.__copy__()

@dataclass
class RangeInfo:
    rays_o: torch.Tensor
    rays_d: torch.Tensor

    range_image: torch.Tensor
    range_direction: torch.Tensor
    range_intensity: torch.Tensor
    mask: torch.Tensor

    sensor_center: torch.Tensor
    lidar2world: torch.Tensor
    range_id: Optional[int] = None
    normalized_time: Optional[torch.Tensor] = None

    def to(self, device: torch.device):
        self.rays_o = self.rays_o.to(device, non_blocking=True)
        self.rays_d = self.rays_d.to(device, non_blocking=True)

        self.range_image = self.range_image.to(device, non_blocking=True)
        self.range_direction = self.range_direction.to(device, non_blocking=True)
        self.range_intensity = self.range_intensity.to(device, non_blocking=True)
        self.mask = self.mask.to(device, non_blocking=True)

        self.sensor_center = self.sensor_center.to(device)
        self.lidar2world = self.lidar2world.to(device)
        if self.normalized_time is not None:
            self.normalized_time = self.normalized_time.to(device, non_blocking=True)
        return self

    def detach(self):
        rays_o = self.rays_o.detach()
        rays_d = self.rays_d.detach()
        sensor_center = self.sensor_center.detach()
        lidar2world = self.lidar2world.detach()

        range_image = self.range_image.detach()
        range_direction = self.range_direction.detach()
        range_intensity = self.range_intensity.detach()
        mask = self.mask.detach()

        new_info = RangeInfo(rays_o=rays_o,
                             rays_d=rays_d,
                             range_image=range_image,
                             range_direction=range_direction,
                             range_intensity=range_intensity,
                             mask=mask,
                             sensor_center=sensor_center,
                             lidar2world=lidar2world,
                             range_id=self.range_id)

        if self.normalized_time is not None:
            new_info.normalized_time = self.normalized_time.detach()

        return new_info

    def __copy__(self):
        new_info = RangeInfo(
            rays_o=self.rays_o.copy(),
            rays_d=self.rays_d.copy(),
            range_image=self.range_image.copy(),
            range_direction=self.range_direction.copy(),
            range_intensity=self.range_intensity.copy(),
            mask=self.mask.copy(),
            sensor_center=self.sensor_center,
            lidar2world=self.lidar2world,
            range_id=self.range_id,
        )

        if self.normalized_time is not None:
            new_info.normalized_time = self.normalized_time

        return new_info

    def copy(self):
        return self.__copy__()

@dataclass
class SampleInfo:
    pixel: Optional[Tuple[ImageInfo, CameraInfo]] = None
    range: Optional[Tuple[RangeInfo]] = None