# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.
import torch
from .ARRD import ARRD_Multi_Level_type2
from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector


@DETECTORS.register_module()
class MyYOLOV3(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(MyYOLOV3, self).__init__(backbone, neck, bbox_head, train_cfg,
                                     test_cfg, pretrained, init_cfg)
        self.decoder = ARRD_Multi_Level_type2()

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
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)
        rec_img = self.decoder(x, s=2)
        rec_loss = self.spatial_contrastive_loss(rec_img, img)

        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        losses['rec_loss'] = 0.25 * rec_loss

        return losses


    def onnx_export(self, img, img_metas):
        """Test function for exporting to ONNX, without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head.forward(x)
        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape

        det_bboxes, det_labels = self.bbox_head.onnx_export(*outs, img_metas)

        return det_bboxes, det_labels

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
