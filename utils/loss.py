from abc import ABCMeta, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.common.registry import Registry
#
from utils.registries import LOSS_REGISTRY
#
from .build_helper import LOSS_REGISTRY


class BaseWeightedLoss(nn.Module, metaclass=ABCMeta):
    """Base class for loss.
    All subclass should overwrite the ``_forward()`` method which returns the
    normal loss without loss weights.
    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

    @abstractmethod
    def _forward(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        """Defines the computation performed at every call.
        Args:
            *args: The positional arguments for the corresponding
                loss.
            **kwargs: The keyword arguments for the corresponding
                loss.
        Returns:
            torch.Tensor: The calculated loss.
        """
        ret = self._forward(*args, **kwargs)
        if isinstance(ret, dict):
            for k in ret:
                if 'loss' in k:
                    ret[k] *= self.loss_weight
        else:
            ret *= self.loss_weight
        return ret


@LOSS_REGISTRY.register()
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.
    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probablity distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.
    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_cfg):
        super().__init__(loss_weight=loss_cfg['LOSS_WEIGHT'])
        self.class_weight = (
            torch.Tensor(loss_cfg['CLASS_WEIGHT'])
            if 'CLASS_WEIGHT' in loss_cfg
            else None
        )

    def _forward(self, outputs, samples, **kwargs):
        """Forward function.
        Args:
            cls_score (torch.Tensor): The class score.
            samples (dict): The ground truth labels.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.
        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        cls_score = outputs['logits']
        label = samples['bin_label_onehot']
        if cls_score.size() == label.size():
            # calculate loss for soft labels

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, (
                'For now, no extra args are supported for soft label, '
                f'but get {kwargs}'
            )

            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                lsm = lsm * self.class_weight.unsqueeze(0).to(cls_score.device)
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0).to(cls_score.device) * label
                )
            else:
                loss_cls = loss_cls.mean()
        else:
            # calculate loss for hard label

            if self.class_weight is not None:
                assert (
                    'weight' not in kwargs
                ), "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)
            loss_cls = F.cross_entropy(cls_score, label, **kwargs)

        return loss_cls


@LOSS_REGISTRY.register()
class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits.
    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Default: None.
    """

    def __init__(self, loss_cfg):
        super().__init__(loss_weight=loss_cfg['LOSS_WEIGHT'])
        self.class_weight = (
            torch.Tensor(loss_cfg['CLASS_WEIGHT'])
            if 'CLASS_WEIGHT' in loss_cfg
            else None
        )

    def _forward(self, outputs, samples, **kwargs):
        """Forward function.
        Args:
            cls_score (torch.Tensor): The class score.
            samples (dict): The ground truth labels.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.
        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        cls_score = outputs['logits']
        label = samples['bin_label_onehot']
        if self.class_weight is not None:
            assert (
                'weight' not in kwargs
            ), "The key 'weight' already exists."
            kwargs['weight'] = self.class_weight.to(cls_score.device)
        loss_cls = F.binary_cross_entropy_with_logits(
            cls_score, label, **kwargs
        )
        return loss_cls


@LOSS_REGISTRY.register()
class MSELoss(BaseWeightedLoss):
    """MSE Loss
    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.

    """

    def __init__(self, loss_cfg):
        super().__init__(loss_weight=loss_cfg['LOSS_WEIGHT'])
        self.mse = nn.MSELoss()

    def _forward(self, pred_mask, gt_mask, **kwargs):  # TODO samples
        loss = self.mse(pred_mask, gt_mask)
        return loss


@LOSS_REGISTRY.register()
class ICCLoss(BaseWeightedLoss):
    """Contrastive Loss
    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Default: 1.0.
    """

    def __init__(self, loss_cfg):
        super().__init__(loss_weight=loss_cfg['CON_LOSS_WEIGHT'])

    # def _forward(self, outputs, samples, **kwargs):  # TODO samples
    #     # size of feature is (b, 1024)
    #     # size of label is (b)
    #     feature = outputs['features']
    #     label = samples['bin_label'].squeeze()
    #     b, feat_dim = feature.shape
    #     device = feature.device
    #
    #     res_label = torch.zeros(label.size(), dtype=label.dtype)
    #     res_label = torch.where(label == 1, 0, 1)
    #     res_label = res_label.type(torch.BoolTensor).cuda()
    #
    #     # print(label, res_label)
    #     pos_feature = torch.masked_select(feature, label)
    #     neg_feature = torch.masked_select(feature, res_label)
    #
    #     # print('pos_fea: ', pos_feature.device)
    #     # print('nge_fea: ', neg_feature.device)
    #     pos_feature = pos_feature.view(-1, feat_dim)
    #     neg_feature = neg_feature.view(-1, feat_dim)
    #
    #     pos_center = torch.mean(pos_feature, dim=0, keepdim=True)
    #
    #     # dis_pos = torch.sum((pos_feature - pos_center)**2) / torch.norm(pos_feature, p=1)
    #     # dis_neg = torch.sum((neg_feature - pos_center)**2) / torch.norm(neg_feature, p=1)
    #     num_p = pos_feature.size(0)
    #     num_n = neg_feature.size(0)
    #     pos_center1 = pos_center.repeat(num_p, 1)
    #     pos_center2 = pos_center.repeat(num_n, 1)
    #     dis_pos = F.cosine_similarity(pos_feature, pos_center1, eps=1e-6)
    #     dis_pos = torch.mean(dis_pos, dim=0)
    #     dis_neg = F.cosine_similarity(neg_feature, pos_center2, eps=1e-6)
    #     dis_neg = torch.mean(dis_neg, dim=0)
    #
    #     loss = dis_pos - dis_neg
    #
    #     return loss

    def _forward(self, outputs, samples, **kwargs):  # TODO samples
        """
            正确的欧式距离版对比损失：类内紧凑、类间分离，物理意义更直观
            :param feature: 特征张量 shape [b, 1024]
            :param label: 标签张量 shape [b], 1=正样本，0=负样本
            :return: loss 标量，优化器最小化该loss即可实现目标
        """
        feature = outputs['features']
        label = samples['bin_label'].squeeze()
        b, feat_dim = feature.shape
        device = feature.device

        # print(label.shape)
        # 1. 生成正负掩码
        pos_mask = label.unsqueeze(1).repeat(1, feat_dim).bool().to(device)
        neg_mask = ~pos_mask

        # 2. 筛选特征
        pos_feat = torch.masked_select(feature, pos_mask).reshape(-1, feat_dim)
        neg_feat = torch.masked_select(feature, neg_mask).reshape(-1, feat_dim)

        # 3. 边界判断
        if len(pos_feat) == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
        if len(neg_feat) == 0:
            pos_center = pos_feat.mean(dim=0, keepdim=True)
            pos_dist = torch.norm(pos_feat - pos_center, p=2, dim=1).mean()
            return pos_dist  # 只约束正样本紧凑

        # 4. 计算正样本中心
        pos_center = pos_feat.mean(dim=0, keepdim=True)

        # 5. 计算欧式距离 (p=2，二范数)
        pos_dist = torch.norm(pos_feat - pos_center.repeat(len(pos_feat), 1), p=2, dim=1).mean()
        neg_dist = torch.norm(neg_feat - pos_center.repeat(len(neg_feat), 1), p=2, dim=1).mean()

        # 6. 核心正确损失公式
        # 损失公式：loss = 正样本平均距离 + (最大距离阈值 - 负样本平均距离)
        # 优化器最小化loss → 正距离↓ 负距离↑，完美实现类内紧凑、类间分离
        max_dist = torch.tensor(feat_dim ** 0.5, device=device)  # 距离阈值，1024维特征的最大理论距离
        loss = pos_dist + (max_dist - neg_dist)

        return loss


@LOSS_REGISTRY.register()
class FocalLoss(BaseWeightedLoss):
    def __init__(self, loss_cfg):
        super().__init__(loss_weight=loss_cfg['CLS_LOSS_WEIGHT'])
        super(FocalLoss, self).__init__()
        self.alpha = loss_cfg['ALPHA'] if 'ALPHA' in loss_cfg.keys() else 1
        self.gamma = loss_cfg['GAMMA'] if 'GAMMA' in loss_cfg.keys() else 2
        self.logits = (
            loss_cfg['LOGITS'] if 'LOGITS' in loss_cfg.keys() else True
        )
        self.reduce = (
            loss_cfg['REDUCE'] if 'REDUCE' in loss_cfg.keys() else True
        )

    def _forward(self, outputs, samples, **kwargs):
        cls_score = outputs['logits']
        label = samples['bin_label_onehot']
        if self.logits:  # TODO
            BCE_loss = F.binary_cross_entropy_with_logits(
                cls_score, label, reduce=False
            )
        else:
            BCE_loss = F.binary_cross_entropy(cls_score, label, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


@LOSS_REGISTRY.register()
class Auxiliary_Loss_v2(BaseWeightedLoss):
    def __init__(self, loss_cfg):
        super().__init__(loss_weight=loss_cfg['AUX_LOSS_WEIGHT'])
        M = loss_cfg['M'] if 'M' in loss_cfg.keys() else 1
        N = loss_cfg['N'] if 'N' in loss_cfg.keys() else 1
        C = loss_cfg['C'] if 'C' in loss_cfg.keys() else 1
        alpha = loss_cfg['ALPHA'] if 'ALPHA' in loss_cfg.keys() else 0.05
        margin = loss_cfg['MARGIN'] if 'MARGIN' in loss_cfg.keys() else 1
        inner_margin = (
            loss_cfg['INNER_MARGIN']
            if 'INNER_MARGIN' in loss_cfg.keys()
            else [0.1, 5]
        )

        self.register_buffer('feature_centers', torch.zeros(M, N))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.num_classes = C
        self.margin = margin
        from M2TR.models.matdd import AttentionPooling

        self.atp = AttentionPooling()
        self.register_buffer('inner_margin', torch.Tensor(inner_margin))

    def _forward(self, feature_map_d, attentions, y):
        B, N, H, W = feature_map_d.size()
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions = F.interpolate(
                attentions, (H, W), mode='bilinear', align_corners=True
            )
        feature_matrix = self.atp(feature_map_d, attentions)
        feature_centers = self.feature_centers
        center_momentum = feature_matrix - feature_centers
        real_mask = (y == 0).view(-1, 1, 1)
        fcts = (
            self.alpha * torch.mean(center_momentum * real_mask, dim=0)
            + feature_centers
        )
        fctsd = fcts.detach()
        if self.training:
            with torch.no_grad():
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        fctsd, torch.distributed.ReduceOp.SUM
                    )
                    fctsd /= torch.distributed.get_world_size()
                self.feature_centers = fctsd
        inner_margin = self.inner_margin[y]
        intra_class_loss = F.relu(
            torch.norm(feature_matrix - fcts, dim=[1, 2])
            * torch.sign(inner_margin)
            - inner_margin
        )
        intra_class_loss = torch.mean(intra_class_loss)
        inter_class_loss = 0
        for j in range(M):
            for k in range(j + 1, M):
                inter_class_loss += F.relu(
                    self.margin - torch.dist(fcts[j], fcts[k]), inplace=False
                )
        inter_class_loss = inter_class_loss / M / self.alpha
        # fmd=attentions.flatten(2)
        # diverse_loss=torch.mean(F.relu(F.cosine_similarity(fmd.unsqueeze(1),fmd.unsqueeze(2),dim=3)-self.margin,inplace=True)*(1-torch.eye(M,device=attentions.device)))
        return intra_class_loss + inter_class_loss, feature_matrix


class Auxiliary_Loss_v1(nn.Module):
    def __init__(self, loss_cfg):
        super().__init__(loss_weight=loss_cfg['loss_weight'])
        M = loss_cfg['M'] if 'M' in loss_cfg.keys() else 1
        N = loss_cfg['N'] if 'N' in loss_cfg.keys() else 1
        C = loss_cfg['N'] if 'N' in loss_cfg.keys() else 1
        alpha = loss_cfg['N'] if 'N' in loss_cfg.keys() else 0.05
        margin = loss_cfg['N'] if 'N' in loss_cfg.keys() else 1
        inner_margin = (
            loss_cfg['inner_margin']
            if 'inner_margin' in loss_cfg.keys()
            else [0.01, 0.02]
        )
        self.register_buffer('feature_centers', torch.zeros(M, N))
        self.register_buffer('alpha', torch.tensor(alpha))
        self.num_classes = C
        self.margin = margin
        from M2TR.models.matdd import AttentionPooling

        self.atp = AttentionPooling()
        self.register_buffer('inner_margin', torch.Tensor(inner_margin))

    def forward(self, feature_map_d, attentions, y):
        B, N, H, W = feature_map_d.size()
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions = F.interpolate(
                attentions, (H, W), mode='bilinear', align_corners=True
            )
        feature_matrix = self.atp(feature_map_d, attentions)
        feature_centers = self.feature_centers.detach()
        center_momentum = feature_matrix - feature_centers
        fcts = self.alpha * torch.mean(center_momentum, dim=0) + feature_centers
        fctsd = fcts.detach()
        if self.training:
            with torch.no_grad():
                if torch.distributed.is_initialized():
                    torch.distributed.all_reduce(
                        fctsd, torch.distributed.ReduceOp.SUM
                    )
                    fctsd /= torch.distributed.get_world_size()
                self.feature_centers = fctsd
        inner_margin = torch.gather(
            self.inner_margin.repeat(B, 1), 1, y.unsqueeze(1)
        )
        intra_class_loss = F.relu(
            torch.norm(feature_matrix - fcts, dim=-1) - inner_margin
        )
        intra_class_loss = torch.mean(intra_class_loss)
        inter_class_loss = 0
        for j in range(M):
            for k in range(j + 1, M):
                inter_class_loss += F.relu(
                    self.margin - torch.dist(fcts[j], fcts[k]), inplace=False
                )
        inter_calss_loss = inter_class_loss / M / self.alpha
        # fmd=attentions.flatten(2)
        # inter_class_loss=torch.mean(F.relu(F.cosine_similarity(fmd.unsqueeze(1),fmd.unsqueeze(2),dim=3)-self.margin,inplace=True)*(1-torch.eye(M,device=attentions.device)))
        return intra_class_loss + inter_class_loss, feature_matrix


@LOSS_REGISTRY.register()
class Multi_attentional_Deepfake_Detection_loss(nn.Module):
    def __init__(self, loss_cfg) -> None:
        super().__init__()
        self.loss_cfg = loss_cfg

    def forward(self, loss_pack, label):
        if 'loss' in loss_pack:
            return loss_pack['loss']
        loss = (
            self.loss_cfg['ENSEMBLE_LOSS_WEIGHT'] * loss_pack['ensemble_loss']
            + self.loss_cfg['AUX_LOSS_WEIGHT'] * loss_pack['aux_loss']
        )
        if self.loss_cfg['AGDA_LOSS_WEIGHT'] != 0:
            loss += (
                self.loss_cfg['AGDA_LOSS_WEIGHT']* loss_pack['AGDA_ensemble_loss']
                + self.loss_cfg['MATCH_LOSS_WEIGHT'] * loss_pack['match_loss']
            )
        return loss

@LOSS_REGISTRY.register()
class Combining_Loss(BaseWeightedLoss):
    def __init__(self, loss_cfg):
        super(Combining_Loss, self).__init__()
        self.cls_loss = FocalLoss(loss_cfg)
        self.contrastive_loss = ICCLoss(loss_cfg)

    def _forward(self, outputs, samples, **kwargs):
        loss = self.cls_loss(outputs, samples) + self.contrastive_loss(outputs, samples)
        return loss

