from omegaconf import OmegaConf
import numpy as np
import os
import time
import wandb
import random
import imageio
import logging
import argparse

import torch
from tools.eval import do_evaluation
from utils.misc import import_str
from utils.backup import backup_project
from utils.logging import MetricLogger, setup_logging
from models.video_utils import render_images, save_videos
from datasets.driving_dataset import DrivingDataset
from datasets.base.data_proto import CameraInfo, ImageInfo, Rays, ImageMasks
from models.sd_pipeline.generative_pipeline import GenerativeReconTrainer


logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

def set_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def setup(args):
    # get config
    cfg = OmegaConf.load(args.config_file)
    
    # parse datasets
    args_from_cli = OmegaConf.from_cli(args.opts)
    if "dataset" in args_from_cli:
        cfg.dataset = args_from_cli.pop("dataset")
        
    assert "dataset" in cfg or "data" in cfg, \
        "Please specify dataset in config or data in config"
        
    if "dataset" in cfg:
        dataset_type = cfg.pop("dataset")
        dataset_cfg = OmegaConf.load(
            os.path.join("configs", "datasets", f"{dataset_type}.yaml")
        )
        # merge data
        cfg = OmegaConf.merge(cfg, dataset_cfg)
    
    # merge cli
    cfg = OmegaConf.merge(cfg, args_from_cli)
    log_dir = os.path.join(args.output_root, args.project, args.run_name)
    
    # update config and create log dir
    cfg.log_dir = log_dir
    os.makedirs(log_dir, exist_ok=True)
    for folder in ["images", "videos", "metrics", "configs_bk", "buffer_maps", "backup"]:
        os.makedirs(os.path.join(log_dir, folder), exist_ok=True)
    
    # setup wandb
    if args.enable_wandb:
        # sometimes wandb fails to init in cloud machines, so we give it several (many) tries
        while (
            wandb.init(
                project=args.project,
                entity=args.entity,
                sync_tensorboard=True,
                settings=wandb.Settings(start_method="fork"),
            )
            is not wandb.run
        ):
            continue
        wandb.run.name = args.run_name
        wandb.run.save()
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.config.update(args)

    # setup random seeds
    set_seeds(cfg.seed)

    global logger
    setup_logging(output=log_dir, level=logging.INFO, time_string=current_time)
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    
    # save config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    saved_cfg_path = os.path.join(log_dir, "config.yaml")
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
        
    # also save a backup copy
    saved_cfg_path_bk = os.path.join(log_dir, "configs_bk", f"config_{current_time}.yaml")
    with open(saved_cfg_path_bk, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    logger.info(f"Full config saved to {saved_cfg_path}, and {saved_cfg_path_bk}")
    
    # Backup codes
    backup_project(
        os.path.join(log_dir, 'backup'), "./", 
        ["configs", "datasets", "models", "utils", "tools"], 
        [".py", ".h", ".cpp", ".cuh", ".cu", ".sh", ".yaml"]
    )
    return cfg

def transformation_struct(image_infos, cam_infos):
    origins = image_infos["origins"]
    viewdirs = image_infos["viewdirs"]
    direction_norm = image_infos["direction_norm"]
    pixel_coords = image_infos["pixel_coords"]
    normed_time = image_infos["normed_time"]
    img_idx = image_infos["img_idx"]
    frame_idx = image_infos["frame_idx"]
    rgb = image_infos["pixels"]
    sky_mask = image_infos["sky_masks"]
    dynamic_mask = image_infos["dynamic_masks"]
    human_mask = image_infos["human_masks"]
    vehicle_mask = image_infos["vehicle_masks"]
    egocar_mask = image_infos["egocar_masks"]
    lidar_depth_map = image_infos["lidar_depth_map"]
    normalized_time = image_infos["normalized_time"]
    unique_img_idx = image_infos["unique_img_idx"]
    sd_image_info = ImageInfo(
        rays=Rays(origins, viewdirs, direction_norm),
        pixels=rgb,
        depth_map=lidar_depth_map,
        masks=ImageMasks(sky_mask, dynamic_mask, human_mask, vehicle_mask, egocar_mask),
        pixel_coords=pixel_coords,
        normalized_time=normalized_time,
        image_index=unique_img_idx,
        frame_index=torch.tensor(frame_idx),
    )
    cam_id = cam_infos["cam_id"]
    cam_name = cam_infos["cam_name"]
    camera_to_world = cam_infos["camera_to_world"]
    height = cam_infos["height"]
    width = cam_infos["width"]
    intrinsics = cam_infos["intrinsics"]
    ego_to_world = cam_infos["ego_to_world"]
    cam_to_ego = cam_infos["cam_to_ego"]
    sd_camera_info = CameraInfo(
        camera_id=cam_id,
        camera_name=cam_name,
        intrinsic=intrinsics,
        camera_to_world=camera_to_world, # Twc
        ego_to_world=ego_to_world, # Tw_ego
        camera_to_ego=cam_to_ego, # Extrinsic, Tego_c
        height=height,
        width=width
    )
    return sd_image_info, sd_camera_info

def reverse_transformation_struct(sd_image_info, sd_camera_info):
    image_infos = {
        "origins": sd_image_info.rays.origins,
        "viewdirs": sd_image_info.rays.viewdirs,
        "direction_norm": sd_image_info.rays.direction_norm,
        "pixel_coords": sd_image_info.pixel_coords,
        "normed_time": sd_image_info.normalized_time,
        "img_idx": sd_image_info.image_index,
        "frame_idx": sd_image_info.frame_index,
        "pixels": sd_image_info.pixels,
        "sky_masks": sd_image_info.masks.sky_mask,
        "dynamic_masks": sd_image_info.masks.dynamic_mask,
        "human_masks": sd_image_info.masks.human_mask,
        "vehicle_masks": sd_image_info.masks.vehicle_mask,
        "egocar_masks": sd_image_info.masks.egocar_mask,
        "lidar_depth_map": sd_image_info.depth_map,
        "normalized_time": sd_image_info.normalized_time,
        "unique_img_idx": sd_image_info.image_index
    }
    cam_infos = {
        "cam_id": sd_camera_info.camera_id,
        "cam_name": sd_camera_info.camera_name,
        "camera_to_world": sd_camera_info.camera_to_world,
        "height": sd_camera_info.height,
        "width": sd_camera_info.width,
        "intrinsics": sd_camera_info.intrinsic,
        "ego_to_world": sd_camera_info.ego_to_world,
        "cam_to_ego": sd_camera_info.camera_to_ego
    }
    return image_infos, cam_infos

# def _prepare_generative_engine_inference_data(shift):

def main(args):
    cfg = setup(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.project_dir = cfg.log_dir

    # build dataset
    dataset = DrivingDataset(data_cfg=cfg.data, project_dir=cfg.project_dir)

    # adapt sd pipeline
    use_model_parallel = cfg.multi_gpu.use_model_parallel
    if use_model_parallel:
        device_recon = torch.device("cuda:0")
        device_gen = torch.device("cuda:1")
        device = device_recon
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device_recon = device
        device_gen = device
    sd_pipeline_flag = True
    if sd_pipeline_flag:
        generative_recon_pipeline = GenerativeReconTrainer(
                cfg=cfg,
                devices=[device_recon, device_gen],
                num_train_images=len(dataset.train_image_set),
            )
    # import pdb; pdb.set_trace()

    # setup trainer
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device
    )
    
    # NOTE: If resume, gaussians will be loaded from checkpoint
    #       If not, gaussians will be initialized from dataset
    if args.resume_from is not None:
        trainer.resume_from_checkpoint(
            ckpt_path=args.resume_from,
            load_only_model=True
        )
        logger.info(
            f"Resuming training from {args.resume_from}, starting at step {trainer.step}"
        )
    else:
        trainer.init_gaussians_from_dataset(dataset=dataset)
        logger.info(
            f"Training from scratch, initializing gaussians from dataset, starting at step {trainer.step}"
        )
    
    if args.enable_viewer:
        # a simple viewer for background visualization
        trainer.init_viewer(port=args.viewer_port)
    
    # define render keys
    render_keys = [
        "gt_rgbs",
        "rgbs",
        "Background_rgbs",
        "Dynamic_rgbs",
        "RigidNodes_rgbs",
        "DeformableNodes_rgbs",
        "SMPLNodes_rgbs",
        # "depths",
        # "Background_depths",
        # "Dynamic_depths",
        # "RigidNodes_depths",
        # "DeformableNodes_depths",
        # "SMPLNodes_depths",
        # "mask"
    ]
    if cfg.render.vis_lidar:
        render_keys.insert(0, "lidar_on_images")
    if cfg.render.vis_sky:
        render_keys += ["rgb_sky_blend", "rgb_sky"]
    if cfg.render.vis_error:
        render_keys.insert(render_keys.index("rgbs") + 1, "rgb_error_maps")
    
    # setup optimizer  
    trainer.initialize_optimizer()
    
    # setup metric logger
    metrics_file = os.path.join(cfg.log_dir, "metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    all_iters = np.arange(trainer.step, trainer.num_iters + 1)
    
    # DEBUG USE
    # do_evaluation(
    #     step=0,
    #     cfg=cfg,
    #     trainer=trainer,
    #     dataset=dataset,
    #     render_keys=render_keys,
    #     args=args,
    # )

    for step in metric_logger.log_every(all_iters, cfg.logging.print_freq):
        #----------------------------------------------------------------------------
        #----------------------------     Validate     ------------------------------
        if step % cfg.logging.vis_freq == 0 and cfg.logging.vis_freq > 0:
            logger.info("Visualizing...")
            vis_timestep = np.linspace(
                0,
                dataset.num_img_timesteps,
                trainer.num_iters // cfg.logging.vis_freq + 1,
                endpoint=False,
                dtype=int,
            )[step // cfg.logging.vis_freq]
            with torch.no_grad():
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                    compute_metrics=True,
                    compute_error_map=cfg.render.vis_error,
                    vis_indices=[
                        vis_timestep * dataset.pixel_source.num_cams + i
                        for i in range(dataset.pixel_source.num_cams)
                    ],
                )
            if args.enable_wandb:
                wandb.log(
                    {
                        "image_metrics/psnr": render_results["psnr"],
                        "image_metrics/ssim": render_results["ssim"],
                        "image_metrics/occupied_psnr": render_results["occupied_psnr"],
                        "image_metrics/occupied_ssim": render_results["occupied_ssim"],
                    }
                )
            vis_frame_dict = save_videos(
                render_results,
                save_pth=os.path.join(
                    cfg.log_dir, "images", f"step_{step}.png"
                ),  # don't save the video
                layout=dataset.layout,
                num_timestamps=1,
                keys=render_keys,
                save_seperate_video=cfg.logging.save_seperate_video,
                num_cams=dataset.pixel_source.num_cams,
                fps=cfg.render.fps,
                verbose=False,
            )
            if args.enable_wandb:
                for k, v in vis_frame_dict.items():
                    wandb.log({"image_rendering/" + k: wandb.Image(v)})
            del render_results
            torch.cuda.empty_cache()
                
        
        #----------------------------------------------------------------------------
        #----------------------------  training step  -------------------------------
        # prepare for training
        trainer.set_train()
        trainer.preprocess_per_train_step(step=step)
        trainer.optimizer_zero_grad() # zero grad
        tmp_print = False
        if sd_pipeline_flag:
            generative_recon_pipeline.run_before_train_step(step)
            if generative_recon_pipeline.generative_scheduler.num_novel_data > 0:
                # import pdb; pdb.set_trace()
                de_img, ref_image, ref_sky_mask, ref_c2w, index, shift_value_name = generative_recon_pipeline.generative_scheduler.get_novel_data()
                from einops import rearrange
                ref_image = rearrange(ref_image, "c h w -> h w c")
                de_img = rearrange(de_img, "c h w -> h w c")
                # shift_value_name 可能是个 BUG 
                old_shift_value_name = f"{generative_recon_pipeline.delta_shift * generative_recon_pipeline.current_shift_level:.1f}"
                print(
                    "step = ", step,
                    "\t in-beforeTrain::save_novel_view_data :",
                    "\t fix_shift = ", shift_value_name,
                    "\t old_shift_value_name = ", old_shift_value_name,
                    "\t delta_shift = ", generative_recon_pipeline.delta_shift,
                    "\t current_shift_level = ", generative_recon_pipeline.current_shift_level,
                    "\t image_index(save) = ", index
                    )

                tmp_print = True
                # import pdb; pdb.set_trace()
                dataset.save_novel_view_data(
                    index,
                    shift_value_name,
                    novel_view_cam_extrinsic=ref_c2w,
                    novel_view_render_image=de_img,
                    novel_view_render_fix_image=ref_image,
                    novel_view_sky_mask=ref_sky_mask,
                )
        
        # get data
        train_step_camera_downscale = trainer._get_downscale_factor()
        image_infos, cam_infos = dataset.train_image_set.next(train_step_camera_downscale)
        for k, v in image_infos.items():
            if isinstance(v, torch.Tensor):
                image_infos[k] = v.cuda(non_blocking=True)
        for k, v in cam_infos.items():
            if isinstance(v, torch.Tensor):
                cam_infos[k] = v.cuda(non_blocking=True)
        
        # forward & backward
        outputs = trainer(image_infos, cam_infos)
        trainer.update_visibility_filter()

        loss_dict = trainer.compute_losses(
            outputs=outputs,
            image_infos=image_infos,
            cam_infos=cam_infos,
        )

        if sd_pipeline_flag:
            # image_infos = image_infos.image_index.item()
            index = image_infos["img_idx"][0,0].item()
            shift_value_name = f"{generative_recon_pipeline.delta_shift * generative_recon_pipeline.current_shift_level:.1f}"
            if tmp_print :
                print(
                    "step = ", step,
                    '\t in-Forward = ',
                    "\t shift_value_name = ", shift_value_name,
                    "\t delta_shift = ", generative_recon_pipeline.delta_shift,
                    "\t current_shift_level = ", generative_recon_pipeline.current_shift_level,
                    "\t images_index = ", index,
                )

            if (step > cfg.joint_training_cfg.start_engine_infer_at) and dataset.exist_novel_view_data(index, shift_value_name):
                nv_image_infos, nv_cam_infos = dataset.load_novel_view_data(index, shift_value_name, image_infos, cam_infos, device_recon)
                # import pdb; pdb.set_trace()
                nv_infos = dataset.get_img_cam_infos(index, shift_value_name)
                nv_infos_flag = False
                if (nv_image_infos is not None) and (nv_cam_infos is not None):
                    nv_infos_flag = True
                    # nv_image_infos = image_infos
                    # nv_cam_infos = cam_infos
                    # nv_cam_infos["camera_to_world"] = nv_infos['c2w']
                    # if True:
                    #     import pdb; pdb.set_trace()

                    # nv_cam_infos["camera_to_world"] 存放在 cpu 上的； 有影响吗
                    nv_outputs = trainer(nv_image_infos, nv_cam_infos)
                    trainer.update_visibility_filter()
                    novel_view_loss_dict = trainer.compute_losses(
                        outputs=nv_outputs,
                        image_infos=nv_image_infos,
                        cam_infos=nv_cam_infos,
                        from_synthesis=True,
                    )
                    del nv_image_infos, nv_cam_infos
                    for k, v in novel_view_loss_dict.items():
                        if k not in loss_dict:
                            loss_dict[k] = 0
                        loss_dict[k] += v
                print('index = ', index, "\t shift_value_name = ", shift_value_name, "\t nv_infos_flag = ", nv_infos_flag)

        # check nan or inf
        for k, v in loss_dict.items():
            if torch.isnan(v).any():
                raise ValueError(f"NaN detected in loss {k} at step {step}")
            if torch.isinf(v).any():
                raise ValueError(f"Inf detected in loss {k} at step {step}")
        trainer.backward(loss_dict)
        
        # after training step
        trainer.postprocess_per_train_step(step=step)
        

        if sd_pipeline_flag:
            sd_train_data = transformation_struct(image_infos, cam_infos)
            get_nvs_render = generative_recon_pipeline.run_after_train_step(step, sd_train_data, outputs, loss_dict)
            # if get_nvs_render or (step % 50 == 0):
            if get_nvs_render:
                image_info, cam_info = sd_train_data
                image_index = image_info.image_index.item()
                if image_info.masks.egocar_mask is not None:
                    egocar_mask = 1.0 - image_info.masks.egocar_mask # 为啥 1.0 减法啊
                else:
                    egocar_mask = torch.ones(image_info.pixels.shape[:2], device=image_info.pixels.device)

                shift = generative_recon_pipeline.delta_shift * generative_recon_pipeline.current_shift_level
                if tmp_print :
                    print(
                        "step = ", step,
                        '\t in-run_after_train::prepare_sd_repair_imgs = ',
                        "\t shift = ", shift,
                        "\t delta_shift = ", generative_recon_pipeline.delta_shift,
                        "\t current_shift_level = ", generative_recon_pipeline.current_shift_level,
                        "\t image_index = ", image_index,
                    )

                if (shift > 0.0) and (not dataset.exist_novel_view_data(image_index, f"{shift:.1f}")):
                    # clone camera info and adjust pose
                    novel_view_cam_info = cam_info.detach().copy()
                    cam_to_ego = novel_view_cam_info.camera_to_ego.clone()
                    cam_to_ego[1, 3] += shift
                    ego_to_world = novel_view_cam_info.ego_to_world
                    c2w = ego_to_world @ cam_to_ego
                    c2w_recon_d = c2w.to(device_recon)

                    # move camera info to device_recon
                    novel_view_cam_info.camera_to_world = c2w_recon_d
                    novel_view_cam_info = novel_view_cam_info.to(device_recon)
                    image_info = image_info.to(device_recon)
                    # 只改了相机信息，没改动图像
                    nv_img_info, nv_cam_info = reverse_transformation_struct(image_info, novel_view_cam_info)
                    trainer.set_eval()
                    with torch.no_grad():
                        nv_outputs = trainer(nv_img_info, nv_cam_info)
                        # TODO :  save novel_view_cam_info  
                        generative_recon_pipeline._prepare_generative_engine_inference_data(
                            index=image_index,
                            shift=f"{shift:.1f}",
                            results=nv_outputs,
                            c2w=c2w,
                            valid_mask=egocar_mask,
                        )
                        dataset.push_img_cam_infos(
                            image_index=image_index,
                            shift_value_name=f"{shift:.1f}",
                            state="seed",
                            c2w=c2w_recon_d,
                            nv_img_info=None,
                            nv_cam_info=None,
                        )
                        del image_info, novel_view_cam_info
                        del nv_img_info, nv_cam_info
                        del nv_outputs
                        del c2w

        #----------------------------------------------------------------------------
        #-------------------------------  logging  ----------------------------------
        with torch.no_grad():
            # cal stats
            metric_dict = trainer.compute_metrics(
                outputs=outputs,
                image_infos=image_infos,
            )
        metric_logger.update(**{"train_metrics/"+k: v.item() for k, v in metric_dict.items()})
        metric_logger.update(**{"train_stats/gaussian_num_" + k: v for k, v in trainer.get_gaussian_count().items()})
        metric_logger.update(**{"losses/"+k: v.item() for k, v in loss_dict.items()})
        metric_logger.update(**{"train_stats/lr_" + group['name']: group['lr'] for group in trainer.optimizer.param_groups})
        if args.enable_wandb:
            wandb.log({k: v.avg for k, v in metric_logger.meters.items()})

        #----------------------------------------------------------------------------
        #----------------------------     Saving     --------------------------------
        do_save = step > 0 and (
            (step % cfg.logging.saveckpt_freq == 0) or (step == trainer.num_iters)
        ) and (args.resume_from is None)
        if do_save:  
            trainer.save_checkpoint(
                log_dir=cfg.log_dir,
                save_only_model=True,
                is_final=step == trainer.num_iters,
            )
        
        #----------------------------------------------------------------------------
        #------------------------    Cache Image Error    ---------------------------
        if (
            step > 0 and trainer.optim_general.cache_buffer_freq > 0
            and step % trainer.optim_general.cache_buffer_freq == 0
        ):
            logger.info("Caching image error...")
            trainer.set_eval()
            with torch.no_grad():
                dataset.pixel_source.update_downscale_factor(
                    1 / dataset.pixel_source.buffer_downscale
                )
                render_results = render_images(
                    trainer=trainer,
                    dataset=dataset.full_image_set,
                )
                dataset.pixel_source.reset_downscale_factor()
                dataset.pixel_source.update_image_error_maps(render_results)

                # save error maps
                merged_error_video = dataset.pixel_source.get_image_error_video(
                    dataset.layout
                )
                imageio.mimsave(
                    os.path.join(
                        cfg.log_dir, "buffer_maps", f"buffer_maps_{step}.mp4"
                    ),
                    merged_error_video,
                    fps=cfg.render.fps,
                )
            logger.info("Done caching rgb error maps")
            
    
    logger.info("Training done!")

    do_evaluation(
        step=step,
        cfg=cfg,
        trainer=trainer,
        dataset=dataset,
        render_keys=render_keys,
        args=args,
    )
    
    if args.enable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)
    
    return step

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train Gaussian Splatting for a single scene")
    parser.add_argument("--config_file", help="path to config file", type=str)
    parser.add_argument("--output_root", default="./work_dirs/", help="path to save checkpoints and logs", type=str)
    
    # eval
    parser.add_argument("--resume_from", default=None, help="path to checkpoint to resume from", type=str)
    parser.add_argument("--render_video_postfix", type=str, default=None, help="an optional postfix for video")    
    
    # wandb logging part
    parser.add_argument("--enable_wandb", action="store_true", help="enable wandb logging")
    parser.add_argument("--entity", default="ziyc", type=str, help="wandb entity name")
    parser.add_argument("--project", default="drivestudio", type=str, help="wandb project name, also used to enhance log_dir")
    parser.add_argument("--run_name", default="omnire", type=str, help="wandb run name, also used to enhance log_dir")
    
    # viewer
    parser.add_argument("--enable_viewer", action="store_true", help="enable viewer")
    parser.add_argument("--viewer_port", type=int, default=8080, help="viewer port")
    
    # misc
    parser.add_argument("opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER)
    
    args = parser.parse_args()
    final_step = main(args)
