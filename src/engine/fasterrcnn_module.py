import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from omegaconf import DictConfig
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead

from src.engine.compute_metrics import MetricsComputer


class FasterRCNNModule(pl.LightningModule):
    """
    Class implementing a Faster R-CNN as a PyTorch Lightning module

    Attributes
    ----------
    model: torchvision.models.detection.fasterrcnn_resnet50_fpn
        The Faster R-CNN model
    learning_rate: float
        The learning rate
    metrics_computer: MetricsComputer
        The metrics computer
    val_f_score_steps: list
        the F-score on the validation data at each step
    val_f_score_epochs: list
        the F-score on the validation data at each epochs

    """

    def __init__(self, cfg: DictConfig = None):
        """
        Init function

        Note that the config is only needed for training.

        Parameters
        ----------
        cfg: DictConfig
            The configuration

        """
        super().__init__()
        self.model = self._get_model(num_classes=2)

        # config is used for training, not for inference
        if cfg is not None:
            self.model.train()
            self.learning_rate = cfg.train.learning_rate

            # specify an IoU threshold here
            self.metrics_computer = MetricsComputer()
            self.val_f_score_steps = []
            self.val_f_score_epochs = []

    def _get_model(
        self, num_classes: int
    ) -> torchvision.models.detection.fasterrcnn_resnet50_fpn:
        """
        Returns the model defined the config params

        """
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights="FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
            weights_backbone="IMAGENET1K_V1",
        )
        # number of different feature map sizes output by the backbone model
        n_features_map_sizes = 5

        # custom anchor generator
        # run tools/data/check_bboxes_aspect_ratio.py to get the numbers
        anchor_generator = AnchorGenerator(
            sizes=tuple([(8, 16, 32) for _ in range(n_features_map_sizes)]),
            aspect_ratios=tuple(
                [(0.6, 0.83, 1.11) for _ in range(n_features_map_sizes)]
            ),
        )
        model.rpn.anchor_generator = anchor_generator

        # 256 because that's the number of features that FPN returns
        model.rpn.head = RPNHead(
            256, anchor_generator.num_anchors_per_location()[0]
        )
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # replace the pre-trained head with a new one (right number of classes)
        model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes
        )
        return model

    def forward(self, images, target=None):
        return self.model(images, target)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self.model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        self.log("train_loss", losses)
        return losses

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        predictions = self.model(images)
        f_score = self.metrics_computer.run_on_batch(predictions, targets)
        self.val_f_score_steps.append(f_score)

    def on_validation_epoch_end(self):
        f_score = np.mean(self.val_f_score_steps)
        self.val_f_score_epochs.append(f_score)
        self.log("val_f_score", f_score)
        self.val_f_score_steps.clear()

    def on_train_end(self):
        """ """
        print("=" * 50)
        print("--- Validation F-scores ---")
        for i, f in enumerate(self.val_f_score_epochs):
            print(f"{i} -> {f}")
        print("=" * 50)

    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=self.learning_rate, momentum=0.9, weight_decay=0.0005
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=0.1
        )
        return [optimizer], [lr_scheduler]
