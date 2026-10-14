import numpy as np
import torch
import torch.nn.functional as F
from timm.utils import accuracy

import utils.checkpoint as cu
import utils.distributed as du
import utils.logging as logging
from utils.build_helper import (
    build_dataloader,
    build_dataset,
    build_loss_fun,
    build_model,
)
from utils.meters import AucMetric, MetricLogger

import matplotlib.pyplot as plt
import time

yellow_green_cmap = ['#FFFF00', '#ADFF2F', '#00FF00']

logger = logging.get_logger(__name__)

def map_visual(output, sample, id):
    map = output["map"]
    logit = output["logits"]
    img = sample["img"]
    map_scaled = F.interpolate(
        map,
        size=(320, 320), mode="bilinear", align_corners=True
        # size=(256, 256), mode="bilinear", align_corners=True
    )
    map_mean = torch.mean(map_scaled, dim=1, keepdim=True)
    map_mean = F.avg_pool2d(map_mean, kernel_size=11, padding=5, stride=1)
    print(map_mean.shape)
    for i in range(10):
        smoothed = map_mean[i, 0, :, :].detach().cpu().numpy()
        fig, ax = plt.subplots(figsize=(8, 8), dpi=32)
        # cur_logit = logit[i, 0].detach().cpu().numpy()
        cur_img = img[i, :, :,:].permute(1, 2, 0).detach().cpu().numpy()
        cur_img = (cur_img - cur_img.min() )/(cur_img.max() - cur_img.min())
        ax.axis('off')
        ax.imshow(cur_img)
        plt.tight_layout(pad=0)  # 去掉画布边距
        # plt.savefig(f"./img/img_{id*10 +i}.jpg", bbox_inches='tight', pad_inches=0)
        ax.axis('off')
        ax.imshow(
            smoothed,
            cmap='coolwarm',
            interpolation='bilinear',
            alpha=0.6, # 核心：热力图透明度
            vmin=None,  # 颜色映射极值（自动用数据min/max）
            vmax=None
        )
        # ax.text(
        #     x=0.1,        # 水平位置（越小越靠左）
        #     y=0.9,        # 垂直位置（越小越靠上）
        #     s=f"{cur_logit :.2f}",  # 你要写的内容
        #     fontsize=50,          # 字体大小（可自由调整）
        #     color="black",          # 字体颜色（支持英文/十六进制 #FF0000）
        #     weight="bold",        # 加粗（不加粗删掉这行）
        #     transform=plt.gca().transAxes  # 固定在左上角，不随数据变化
        # )
        plt.tight_layout(pad=0)  # 去掉画布边距
        plt.savefig(f"./result_map/map_{id* 10 + i}.jpg", bbox_inches='tight', pad_inches=0)
        plt.clf()
    plt.close()

@torch.no_grad()
def perform_test(
    test_loader, model, cfg, cur_epoch=None, writer=None, mode='Test'
):
    criterion = build_loss_fun(cfg)

    metric_logger = MetricLogger(delimiter="  ")
    auc_metrics = AucMetric(cfg['NUM_GPUS'])
    header = mode + ':'

    model.eval()

    # 收集视频级预测所需的数据
    video_predictions = {}  # {video_id: {'preds': [], 'labels': []}}

    for id, samples in enumerate(metric_logger.log_every(test_loader, 10, header)):
        # 将张量移动到 CUDA，字符串类型（如 video_id）保持在 CPU
        samples = {
            key: sample.cuda(non_blocking=True) if isinstance(sample, torch.Tensor) else sample
            for key, sample in samples.items()
        }
        with torch.cuda.amp.autocast(enabled=cfg['AMP_ENABLE']):
            start = time.time()
            outputs = model(samples)
            end = time.time()
            loss = criterion(outputs, samples)
            # print(f"执行时间：{end - start:.6f} 秒")
            map_visual(outputs, samples, id)
            # outputs['logits'][:, 0] += 0.15
            # outputs['logits'][:, 1] -= 0.15
            preds = F.softmax(outputs['logits'], dim=1)[:, 1]
            auc_metrics.update(samples['bin_label'].squeeze(dim=1), preds)
            
            # 收集视频级预测数据
            video_ids = samples.get('video_id', [])
            if video_ids:
                # video_id 是字符串列表，需要保留原始值
                preds_cpu = preds.detach().cpu().numpy()
                labels_cpu = samples['bin_label'].squeeze(dim=1).detach().cpu().numpy()
                video_ids_cpu = [vid for vid in video_ids]
                
                for vid, pred, label in zip(video_ids_cpu, preds_cpu, labels_cpu):
                    if vid not in video_predictions:
                        video_predictions[vid] = {'preds': [], 'labels': []}
                    video_predictions[vid]['preds'].append(pred)
                    video_predictions[vid]['labels'].append(label)
                    
        acc1 = accuracy(
            outputs['logits'], samples['bin_label'], topk=(1,)
        )
        batch_size = samples['img'].shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1[0].item(), n=batch_size)

    auc_metrics.synchronize_between_processes()
    metric_logger.synchronize_between_processes()
    
    # 计算视频级准确率
    video_acc, video_pred_details = compute_video_level_accuracy(video_predictions)
    
    if writer and cur_epoch is not None:
        writer.add_scalar(
            'acc1', metric_logger.acc1.global_avg, global_step=cur_epoch
        )
        writer.add_scalar('auc', auc_metrics.auc, global_step=cur_epoch)
        writer.add_scalar('video_acc', video_acc, global_step=cur_epoch)

    logger.info(
        '* Acc@1 {top1.global_avg:.3f} Auc {auc:.3f} loss {losses.global_avg:.3f}'.format(
            top1=metric_logger.acc1,
            auc=auc_metrics.auc,
            losses=metric_logger.loss,
        )
    )
    logger.info(
        '* Video-level Accuracy: {video_acc:.3f} (based on {num_videos} videos)'.format(
            video_acc=video_acc,
            num_videos=len(video_predictions),
        )
    )
    
    result = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    result['video_acc'] = video_acc
    result['video_count'] = len(video_predictions)
    return result


def compute_video_level_accuracy(video_predictions, threshold=0.5):
    """
    计算视频级准确率：
    1. 对每个视频的所有帧预测概率取均值
    2. 使用阈值（默认0.5）判断视频是真实还是伪造
    3. 比较视频级预测与视频级标签（取所有帧标签的众数）
    
    Args:
        video_predictions: dict, {video_id: {'preds': [], 'labels': []}}
        threshold: float, 概率阈值，大于等于此值为伪造(1)
    
    Returns:
        video_acc: float, 视频级准确率
        video_pred_details: dict, 每个视频的预测详情
    """
    correct = 0
    total = len(video_predictions)
    video_pred_details = {}
    
    for video_id, data in video_predictions.items():
        preds = np.array(data['preds'])
        labels = np.array(data['labels'])
        
        # 视频级预测：取所有帧概率的均值
        video_pred_prob = np.mean(preds)
        video_pred = 1 if video_pred_prob >= threshold else 0
        
        # 视频级标签：取所有帧标签的众数（同一视频的标签应该一致）
        unique_labels, counts = np.unique(labels, return_counts=True)
        video_label = int(unique_labels[np.argmax(counts)])
        
        # 统计正确数
        if video_pred == video_label:
            correct += 1
        
        video_pred_details[video_id] = {
            'pred_prob': video_pred_prob,
            'pred_label': video_pred,
            'true_label': video_label,
            'frame_count': len(preds),
            'correct': video_pred == video_label,
        }
    
    video_acc = correct / total if total > 0 else 0.0
    return video_acc, video_pred_details


def test(local_rank, num_proc, init_method, shard_id, num_shards, backend, cfg):
    world_size = num_proc * num_shards
    rank = shard_id * num_proc + local_rank
    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    except:
        pass
    torch.cuda.set_device(local_rank)
    du.init_distributed_training(cfg)
    np.random.seed(cfg['RNG_SEED'])
    torch.manual_seed(cfg['RNG_SEED'])

    logging.setup_logging(cfg, mode='test')

    model = build_model(cfg)
    cu.load_test_checkpoint(cfg, model)

    test_dataset = build_dataset('test', cfg)
    test_loader = build_dataloader(test_dataset, 'test', cfg)

    logger.info("Testing model for {} iterations".format(len(test_loader)))

    test_stats = perform_test(test_loader, model, cfg)
    logger.info(
        f"Accuracy of the network on the {len(test_dataset)} test images: {test_stats['acc1']:.1f}%"
    )
