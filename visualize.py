try:
    from vis import save_occ, save_gaussian, save_gaussian_topdown
except:
    print('Load Occupancy Visualization Tools Failed.')
import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist

from PIL import Image
from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")


def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # global settings
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # load config
    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    # init DDP
    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))  # number of nodes
        rank = int(os.environ.get("RANK", 0))  # node id
        gpus = torch.cuda.device_count()  # gpus per node
        print(f"tcp://{ip}:{port}")
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1
    
    writer = None
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # build model
    import model
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    n_parameters = sum(p.numel() for p in my_model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')
    if distributed:
        if cfg.get('syncBN', True):
            my_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(my_model)
            logger.info('converted sync bn.')

        find_unused_parameters = cfg.get('find_unused_parameters', False)
        ddp_model_module = torch.nn.parallel.DistributedDataParallel
        my_model = ddp_model_module(
            my_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        raw_model = my_model.module
    else:
        my_model = my_model.cuda()
        raw_model = my_model
    logger.info('done ddp model')

    cfg.val_dataset_config.update({
        "vis_indices": args.vis_index,
        "num_samples": args.num_samples,
        "vis_scene_index": args.vis_scene_index})

    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        val_only=True)
    
    # resume and load
    cfg.resume_from = ''
    if osp.exists(osp.join(args.work_dir, 'latest.pth')):
        cfg.resume_from = osp.join(args.work_dir, 'latest.pth')
    if args.resume_from:
        cfg.resume_from = args.resume_from
    
    logger.info('resume from: ' + cfg.resume_from)
    logger.info('work dir: ' + args.work_dir)

    if cfg.resume_from and osp.exists(cfg.resume_from):
        map_location = 'cpu'
        ckpt = torch.load(cfg.resume_from, map_location=map_location)
        try:
            # raw_model.load_state_dict(ckpt['state_dict'], strict=True)
            raw_model.load_state_dict(ckpt.get('state_dict', ckpt), strict=True)
        except:
            os.system(f"python modify_weight.py --work-dir {args.work_dir} --epoch {args.epoch}")
            cfg.resume_from = os.path.join(args.work_dir, f"epoch_{args.epoch}_mod.pth")
            ckpt = torch.load(cfg.resume_from, map_location=map_location)
            raw_model.load_state_dict(ckpt['state_dict'], strict=True)
        print(f'successfully resumed.')
    elif cfg.load_from:
        ckpt = torch.load(cfg.load_from, map_location='cpu')
        if 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
        print(raw_model.load_state_dict(state_dict, strict=False))
        
    print_freq = cfg.print_freq
    from misc.metric_util import MeanIoU
    miou_metric = MeanIoU(
        list(range(1, 17)),
        17, #17,
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
         True, 17, filter_minmax=False)
    miou_metric.reset()

    my_model.eval()
    os.environ['eval'] = 'true'
    if args.vis_occ or args.vis_gaussian or args.vis_gaussian_topdown:
        save_dir = os.path.join(args.work_dir, f'vis_ep{args.epoch}')
        os.makedirs(save_dir, exist_ok=True)
    if args.model_type == "base":
        draw_gaussian_params = dict(
            scalar = 1.5,
            ignore_opa = False,
            filter_zsize = False
        )
    elif args.model_type == "prob":
        draw_gaussian_params = dict(
            scalar = 2.0,
            ignore_opa = True,
            filter_zsize = True
        )

    with torch.no_grad():
        for i_iter_val, data in enumerate(val_dataset_loader):
            
            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            input_imgs = data.pop('img')
            ori_imgs = data.pop('ori_img')
            for i in range(ori_imgs.shape[-1]):
                ori_img = ori_imgs[0, ..., i].cpu().numpy()
                ori_img = ori_img[..., [2, 1, 0]]
                ori_img = Image.fromarray(ori_img.astype(np.uint8))
                ori_img.save(os.path.join(save_dir, f'{i_iter_val}_image_{i}.png'))
            
            # breakpoint()
            result_dict = my_model(imgs=input_imgs, metas=data)
            for idx, pred in enumerate(result_dict['final_occ']):
                pred_occ = pred
                gt_occ = result_dict['sampled_label'][idx]
                occ_shape = [200, 200, 16]
                if args.vis_gaussian_topdown:
                    save_gaussian_topdown(
                        save_dir,
                        result_dict['anchor_init'],
                        result_dict['gaussians'],
                        f'val_{i_iter_val}_topdown'
                    )
                if args.vis_occ:
                    save_occ(
                        save_dir,
                        pred_occ.reshape(1, *occ_shape),
                        f'val_{i_iter_val}_pred',
                        True, 0, dataset=args.dataset)
                    save_occ(
                        save_dir,
                        gt_occ.reshape(1, *occ_shape),
                        f'val_{i_iter_val}_gt',
                        True, 0, dataset=args.dataset)
                if args.vis_gaussian:
                    save_gaussian(
                        save_dir,
                        result_dict['gaussian'],
                        f'val_{i_iter_val}_gaussian',
                        **draw_gaussian_params)
                miou_metric._after_step(pred_occ, gt_occ)
            
            if i_iter_val % print_freq == 0 and local_rank == 0:
                logger.info('[EVAL] Iter %5d'%(i_iter_val))
                    
    miou, iou2 = miou_metric._after_epoch()
    logger.info(f'mIoU: {miou}, iou2: {iou2}')
    miou_metric.reset()
    
    if writer is not None:
        writer.close()
        

if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    parser.add_argument('--work-dir', type=str, default='./out/tpv_lidarseg')
    parser.add_argument('--resume-from', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--vis-occ', action='store_true', default=False)
    parser.add_argument('--vis-gaussian', action='store_true', default=False)
    parser.add_argument('--vis_gaussian_topdown', action='store_true', default=False)
    parser.add_argument('--vis-index', type=int, nargs='+', default=[])
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--vis_scene_index', type=int, default=-1)
    parser.add_argument('--vis-scene', action='store_true', default=False)
    parser.add_argument('--epoch', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='nusc')
    parser.add_argument('--model-type', type=str, default="base", choices=["base", "prob"])
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus
    print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)
