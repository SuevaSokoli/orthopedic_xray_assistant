import torch
import torch.nn as nn
import torchvision.models as models
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import BODY_PARTS, CONDITIONS

# ── Model Definition ───────────────────────────────────────
class OrthoResNet50(nn.Module):
    """
    ResNet50-based model for orthopedic X-ray analysis.
    Uses transfer learning from ImageNet weights.
    Has two output heads:
        - body_part_head: predicts which body part (7 classes)
        - condition_head: predicts normal vs abnormal (2 classes)
    """
    def __init__(self, num_body_parts=7, num_conditions=2, pretrained=True):
        super(OrthoResNet50, self).__init__()

        # Load pretrained ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)

        # Remove the final classification layer
        # We keep everything up to the last pooling layer
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])

        # ResNet50 outputs 2048 features after the pooling layer
        self.feature_dim = 2048

        # Freeze early layers (we only train the last layers)
        # This speeds up training significantly
        self._freeze_early_layers()

        # ── Output heads ──────────────────────────────────
        # Head 1: predict body part
        self.body_part_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_body_parts)
        )

        # Head 2: predict condition (normal/abnormal)
        self.condition_head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_conditions)
        )

    def _freeze_early_layers(self):
        """
        Freeze the first 6 layers of ResNet50.
        These layers detect basic features (edges, textures)
        that are already well-learned from ImageNet.
        """
        children = list(self.feature_extractor.children())
        for layer in children[:6]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x: input image tensor (batch_size, 3, 224, 224)
        Returns:
            body_part_logits: raw scores for each body part
            condition_logits: raw scores for each condition
        """
        # Extract features
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # flatten

        # Pass through both heads
        body_part_logits = self.body_part_head(features)
        condition_logits = self.condition_head(features)

        return body_part_logits, condition_logits


def build_model(pretrained=True):
    """
    Helper function to build and return the model.
    """
    model = OrthoResNet50(
        num_body_parts=len(BODY_PARTS),
        num_conditions=len(CONDITIONS),
        pretrained=pretrained
    )
    return model


def count_parameters(model):
    """
    Counts total and trainable parameters in the model.
    Useful for understanding model complexity.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable