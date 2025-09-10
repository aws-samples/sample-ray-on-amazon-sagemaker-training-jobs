from typing import Dict
import numpy as np
import torch
from torchvision import models
from torchvision.models import ResNet152_Weights


class ResnetModel:
    def __init__(self):
        self.weights = ResNet152_Weights.IMAGENET1K_V1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet152(weights=self.weights).to(self.device)
        self.model.eval()

    def __call__(self, batch: Dict[str, np.ndarray]):
        """Run inference on a batch of images."""
        # Convert to tensor and move to device
        images = torch.stack([torch.tensor(img) for img in batch["transformed_image"]])
        images = images.to(self.device)

        with torch.no_grad():
            predictions = self.model(images)
            # Get top prediction for each image
            predicted_classes = torch.argmax(predictions, dim=1)

        return {
            "predictions": predicted_classes.cpu().numpy(),
            "scores": torch.max(torch.softmax(predictions, dim=1), dim=1)[0]
            .cpu()
            .numpy(),
        }
