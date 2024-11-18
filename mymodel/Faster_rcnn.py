import torch
import torch.nn as nn
from .ARRD import ARRD_Multi_Level
from mmdet.models.builder import DETECTORS, build_backbone, build_detector
from mmdet.models.detectors import TwoStageDetector,FasterRCNN
@DETECTORS.register_module()
class MyFasterRCNN(TwoStageDetector):
    """Implementation of `Faster R-CNN <https://arxiv.org/abs/1506.01497>`_"""

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(MyFasterRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        self.decoder = ARRD_Multi_Level()
        self.rec_loss = nn.L1Loss()

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):

        x = self.extract_feat(img)
        rec_img = self.decoder(x)
        rec_loss = self.spatial_contrastive_loss(rec_img, img)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        losses['rec_loss'] = 0.25 * rec_loss

        return losses

    def GlobalkMaxPooling(self, x, k):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x, _ = x.topk(k, dim=2)
        x = torch.mean(x, dim=2).reshape(x.shape[0], x.shape[1], 1, 1)
        return x

    def gram_matrix(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)
        xT = x.permute(0, 2, 1)
        g = torch.bmm(x, xT)
        return g

    def instanceNorm(self, x):
        eps = 1e-5
        mu = torch.mean(x, dim=(2, 3), keepdim=True)
        var = torch.mean(torch.square(x - mu), dim=(2, 3), keepdim=True)
        x_normalized = (x - mu) / torch.sqrt(var + eps)
        return x_normalized

    def instanceL2Norm(self, x):
        eps = 1e-5
        norm = x.norm(dim=(2, 3), keepdim=True)
        x_normalized = x / (norm + eps)
        return x_normalized

    def covariance_constrastive(self, x1, x2):
        b, c, h, w = x1.shape
        x1 = self.instanceL2Norm(x1)
        x2 = self.instanceL2Norm(x2)
        mask = 1 - torch.eye(c).to(x1.device)
        g1 = self.gram_matrix(x1)
        g2 = self.gram_matrix(x2)
        var = torch.square(g1 - g2) * mask.unsqueeze(0)
        var = var.unsqueeze(1)
        k = c // 16
        loss = self.GlobalkMaxPooling(var, k) * 0.5
        return loss.mean()

    def spatial_contrastive_loss(self, x1, x2):
        b, c, h, w = x1.shape
        var = torch.square(x1 - x2)
        var = torch.mean(var, dim=1, keepdim=True)
        # margin
        var[var < 0.01] = 0.0

        k = (h // 16) * (w // 16)
        loss = self.GlobalkMaxPooling(var, k)
        return loss.mean()


