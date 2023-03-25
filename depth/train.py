# ------------------------------------------------------------------------------
# The code is from GLPDepth (https://github.com/vinvino02/GLPDepth).
# For non-commercial purpose only (research, evaluation etc).
# -----------------------------------------------------------------------------

import os
import cv2
import numpy as np
from datetime import datetime

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from models_depth.model import VPDDepth
from models_depth.optimizer import build_optimizers
import utils_depth.metrics as metrics
from utils_depth.criterion import SiLogLoss
import utils_depth.logging as logging

from dataset.base_dataset import get_dataset
from configs.train_options import TrainOptions
import glob
import utils

metric_name = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log',
               'log10', 'silog']


def load_model(ckpt, model, optimizer=None):
    ckpt_dict = torch.load(ckpt, map_location='cpu')
    # keep backward compatibility
    if 'model' not in ckpt_dict and 'optimizer' not in ckpt_dict:
        state_dict = ckpt_dict
    else:
        state_dict = ckpt_dict['model']
    weights = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            weights[key[len('module.'):]] = value
        else:
            weights[key] = value

    model.load_state_dict(weights)

    if optimizer is not None:
        optimizer_state = ckpt_dict['optimizer']
        optimizer.load_state_dict(optimizer_state)


def main():
    opt = TrainOptions()
    args = opt.initialize().parse_args()
    print(args)

    utils.init_distributed_mode_simple(args)
    print(args)
    device = torch.device(args.gpu)

    pretrain = args.pretrained.split('.')[0]
    maxlrstr = str(args.max_lr).replace('.', '')
    minlrstr = str(args.min_lr).replace('.', '')
    layer_decaystr = str(args.layer_decay).replace('.', '')
    weight_decaystr = str(args.weight_decay).replace('.', '')
    num_filter = str(args.num_filters[0]) if args.num_deconv > 0 else ''
    num_kernel = str(args.deconv_kernels[0]) if args.num_deconv > 0 else ''
    name = [args.dataset, str(args.batch_size), pretrain.split('/')[-1], 'deconv'+str(args.num_deconv), \
        str(num_filter), str(num_kernel), str(args.crop_h), str(args.crop_w), maxlrstr, minlrstr, \
        layer_decaystr, weight_decaystr, str(args.epochs)]
    if args.exp_name != '':
        name.append(args.exp_name)

    exp_name = '_'.join(name)
    print('This experiments: ', exp_name)

    # Logging
    if args.rank == 0:
        exp_name = '%s_%s' % (datetime.now().strftime('%m%d'), exp_name)
        log_dir = os.path.join(args.log_dir, exp_name)
        logging.check_and_make_dirs(log_dir)
        writer = SummaryWriter(logdir=log_dir)
        log_txt = os.path.join(log_dir, 'logs.txt')  
        logging.log_args_to_txt(log_txt, args)

        global result_dir
        result_dir = os.path.join(log_dir, 'results')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
    else:
        log_txt = None
        log_dir = None
        
    model = VPDDepth(args=args)

    # CPU-GPU agnostic settings
    
    cudnn.benchmark = True
    model.to(device)
    model_without_ddp = model
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)


    # Dataset setting
    dataset_kwargs = {'dataset_name': args.dataset, 'data_path': args.data_path}
    dataset_kwargs['crop_size'] = (args.crop_h, args.crop_w)

    train_dataset = get_dataset(**dataset_kwargs)
    val_dataset = get_dataset(**dataset_kwargs, is_train=False)

    sampler_train = torch.utils.data.DistributedSampler(
        train_dataset, num_replicas=utils.get_world_size(), rank=args.rank, shuffle=True, 
    )

    sampler_val = torch.utils.data.DistributedSampler(
            val_dataset, num_replicas=utils.get_world_size(), rank=args.rank, shuffle=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               sampler=sampler_train, num_workers=args.workers, 
                                               pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=sampler_val,
                                             pin_memory=True)

    # Training settings
    criterion_d = SiLogLoss()

    optimizer = build_optimizers(model, dict(type='AdamW', lr=args.max_lr, betas=(0.9, 0.999), weight_decay=args.weight_decay,
                constructor='LDMOptimizerConstructor',
                paramwise_cfg=dict(layer_decay_rate=args.layer_decay, no_decay_names=['relative_position_bias_table', 'rpe_mlp', 'logit_scale'])))

    start_ep = 1
    if args.resume_from:
        load_model(args.resume_from, model.module, optimizer)
        strlength = len('_model.ckpt')
        resume_ep = int(args.resume_from[-strlength-2:-strlength])
        print(f'resumed from epoch {resume_ep}, ckpt {args.resume_from}')
        start_ep = resume_ep + 1
    if args.auto_resume:
        ckpt_list = glob.glob(f'{log_dir}/epoch_*_model.ckpt')
        strlength = len('_model.ckpt')
        idx = [ckpt[-strlength-2:-strlength] for ckpt in ckpt_list]
        if len(idx) > 0:
            idx.sort(key=lambda x: -int(x))
            ckpt = f'{log_dir}/epoch_{idx[0]}_model.ckpt'
            load_model(ckpt, model.module, optimizer)
            resume_ep = int(idx[0])
            print(f'resumed from epoch {resume_ep}, ckpt {ckpt}')
            start_ep = resume_ep + 1

    global global_step
    iterations = len(train_loader)
    global_step = iterations * (start_ep - 1)

    best_rmse = 1000

    # Perform experiment
    for epoch in range(start_ep, args.epochs + 1):
        print('\nEpoch: %03d - %03d' % (epoch, args.epochs))
        loss_train = train(train_loader, model, criterion_d, log_txt, optimizer=optimizer, 
                           device=device, epoch=epoch, args=args)
        if args.rank == 0:
            writer.add_scalar('Training loss', loss_train, epoch)

        if epoch % args.val_freq == 0:
            results_dict, loss_val = validate(val_loader, model, criterion_d, 
                                              device=device, epoch=epoch, args=args)
            if args.rank == 0:
                writer.add_scalar('Val loss', loss_val, epoch)

                result_lines = logging.display_result(results_dict)
                if args.kitti_crop:
                    print("\nCrop Method: ", args.kitti_crop)
                print(result_lines)

                with open(log_txt, 'a') as txtfile:
                    txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
                    txtfile.write(result_lines)                

                for each_metric, each_results in results_dict.items():
                    writer.add_scalar(each_metric, each_results, epoch)

        if args.rank == 0:
            if args.save_model:
                torch.save(
                    {
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict()
                    },
                    os.path.join(log_dir, 'last.ckpt'))
            
            if results_dict['rmse'] < best_rmse:
                best_rmse = results_dict['rmse']
                torch.save(
                    {
                        'model': model_without_ddp.state_dict(),
                    },
                    os.path.join(log_dir, 'best.ckpt'))



def train(train_loader, model, criterion_d, log_txt, optimizer, device, epoch, args):    
    global global_step
    model.train()
    if args.rank == 0:
        depth_loss = logging.AverageMeter()
    half_epoch = args.epochs // 2
    iterations = len(train_loader)
    result_lines = []
    for batch_idx, batch in enumerate(train_loader):      
        global_step += 1

        if global_step < iterations * half_epoch:
            current_lr = (args.max_lr - args.min_lr) * (global_step /
                                            iterations/half_epoch) ** 0.9 + args.min_lr
        else:
            current_lr = max(args.min_lr, (args.min_lr - args.max_lr) * (global_step /
                                            iterations/half_epoch - 1) ** 0.9 + args.max_lr)

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr*param_group['lr_scale']

        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)

        preds = model(input_RGB, class_ids=batch['class_id'])

        optimizer.zero_grad()
        loss_d = criterion_d(preds['pred_d'].squeeze(dim=0), depth_gt)

        if args.rank == 0:
            depth_loss.update(loss_d.item(), input_RGB.size(0))
        loss_d.backward()
        
        if args.rank == 0:
            if args.pro_bar:
                logging.progress_bar(batch_idx, len(train_loader), args.epochs, epoch,
                                    ('Depth Loss: %.4f (%.4f)' %
                                    (depth_loss.val, depth_loss.avg)))

            if batch_idx % args.print_freq == 0:
                result_line = 'Epoch: [{0}][{1}/{2}]\t' \
                    'Loss: {loss}, LR: {lr}\n'.format(
                        epoch, batch_idx, iterations,
                        loss=depth_loss.avg, lr=current_lr
                    )
                result_lines.append(result_line)
                print(result_line)
        optimizer.step()
    
    if args.rank == 0:
        with open(log_txt, 'a') as txtfile:
            txtfile.write('\nEpoch: %03d - %03d' % (epoch, args.epochs))
            for result_line in result_lines:
                txtfile.write(result_line)   

    return loss_d


def validate(val_loader, model, criterion_d, device, epoch, args):

    if args.rank == 0:
        depth_loss = logging.AverageMeter()
    model.eval()

    ddp_logger = utils.MetricLogger()

    result_metrics = {}
    for metric in metric_name:
        result_metrics[metric] = 0.0

    for batch_idx, batch in enumerate(val_loader):
        input_RGB = batch['image'].to(device)
        depth_gt = batch['depth'].to(device)
        filename = batch['filename'][0]
        class_id = batch['class_id']

        with torch.no_grad():
            if args.shift_window_test:
                bs, _, h, w = input_RGB.shape
                assert w > h and bs == 1
                interval_all = w - h
                interval = interval_all // (args.shift_size-1)
                sliding_images = []
                sliding_masks = torch.zeros((bs, 1, h, w), device=input_RGB.device)
                class_ids = []
                for i in range(args.shift_size):
                    sliding_images.append(input_RGB[..., :, i*interval:i*interval+h])
                    sliding_masks[..., :, i*interval:i*interval+h] += 1
                    class_ids.append(class_id)
                input_RGB = torch.cat(sliding_images, dim=0)
                class_ids = torch.cat(class_ids, dim=0)
            if args.flip_test:
                input_RGB = torch.cat((input_RGB, torch.flip(input_RGB, [3])), dim=0)
                class_ids = torch.cat((class_ids, class_ids), dim=0)
            pred = model(input_RGB, class_ids=class_ids)
        pred_d = pred['pred_d']
        if args.flip_test:
            batch_s = pred_d.shape[0]//2
            pred_d = (pred_d[:batch_s] + torch.flip(pred_d[batch_s:], [3]))/2.0
        if args.shift_window_test:
            pred_s = torch.zeros((bs, 1, h, w), device=pred_d.device)
            for i in range(args.shift_size):
                pred_s[..., :, i*interval:i*interval+h] += pred_d[i:i+1]
            pred_d = pred_s/sliding_masks

        pred_d = pred_d.squeeze()
        depth_gt = depth_gt.squeeze()

        loss_d = criterion_d(pred_d.squeeze(), depth_gt)

        ddp_logger.update(loss_d=loss_d.item())

        if args.rank == 0:
            depth_loss.update(loss_d.item(), input_RGB.size(0))

        pred_crop, gt_crop = metrics.cropping_img(args, pred_d, depth_gt)
        computed_result = metrics.eval_depth(pred_crop, gt_crop)

        if args.rank == 0:
            save_path = os.path.join(result_dir, filename)

            if save_path.split('.')[-1] == 'jpg':
                save_path = save_path.replace('jpg', 'png')

            if args.save_result:
                if args.dataset == 'kitti':
                    pred_d_numpy = pred_d.cpu().numpy() * 256.0
                    cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
                                [cv2.IMWRITE_PNG_COMPRESSION, 0])
                else:
                    pred_d_numpy = pred_d.cpu().numpy() * 1000.0
                    cv2.imwrite(save_path, pred_d_numpy.astype(np.uint16),
                                [cv2.IMWRITE_PNG_COMPRESSION, 0])
                    
        if args.rank == 0:
            loss_d = depth_loss.avg
            if args.pro_bar:
                logging.progress_bar(batch_idx, len(val_loader), args.epochs, epoch)

        ddp_logger.update(**computed_result)
        for key in result_metrics.keys():
            result_metrics[key] += computed_result[key]

    # for key in result_metrics.keys():
    #     result_metrics[key] = result_metrics[key] / (batch_idx + 1)

    ddp_logger.synchronize_between_processes()

    for key in result_metrics.keys():
        result_metrics[key] = ddp_logger.meters[key].global_avg

    loss_d = ddp_logger.meters['loss_d'].global_avg

    return result_metrics, loss_d


if __name__ == '__main__':
    main()
