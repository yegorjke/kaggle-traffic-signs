from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from dataloaders.traffic_sign_loader import (
    TRAFFIC_SIGN_SAVE_MODEL_DIR,
    TRAFFIC_SIGN_SAVE_MODEL_FILENAME,
    getLabelsNumber,
)
from helpers.helpers import exportOnnxModel

if __name__ == "__main__":
    numLabels = getLabelsNumber()
    modelPath = TRAFFIC_SIGN_SAVE_MODEL_DIR / f"resnet18_{TRAFFIC_SIGN_SAVE_MODEL_FILENAME}"

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, numLabels)
    model.load_state_dict(torch.load(modelPath, map_location="cpu", weights_only=True))

    onnxFilename = TRAFFIC_SIGN_SAVE_MODEL_FILENAME[: TRAFFIC_SIGN_SAVE_MODEL_FILENAME.index(".")]
    exportOnnxModel(model, (3, 64, 64), Path.cwd() / ".saved" / f"resnet18_{onnxFilename}.onnx")
