import argparse

import albumentations as A
import matplotlib.pyplot as plt

# import albumentations
# import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from torchvision import models

from dataloaders.traffic_sign_loader import (
    TRAFFIC_SIGN_DATASET_PATH,
    TRAFFIC_SIGN_SAVE_MODEL_DIR,
    TRAFFIC_SIGN_SAVE_MODEL_FILENAME,
    TrafficSignDataset,
)
from helpers.helpers import getDevice, meanArray, stdArray, transposeToNumpyImage

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference written in PyTorch")
    parser.add_argument("--b", default=1, type=int, help="number of batch")

    args = parser.parse_args()
    batchNumber = 1 if args.b <= 0 else args.b

    device = getDevice()

    trainingLabels = TrafficSignDataset.getLabels(TRAFFIC_SIGN_DATASET_PATH / "labels.csv", hasHeader=True)

    modelPath = TRAFFIC_SIGN_SAVE_MODEL_DIR / ("resnet18_" + TRAFFIC_SIGN_SAVE_MODEL_FILENAME)

    inferenceModel = models.resnet18(weights=None)
    inferenceModel.fc = nn.Linear(inferenceModel.fc.in_features, len(trainingLabels))

    inferenceModel.load_state_dict(torch.load(modelPath, weights_only=True))
    inferenceModel = inferenceModel.to(device)
    inferenceModel.eval()

    testTransforms = A.Compose(
        [
            A.Resize(64, 64),
            A.Normalize(mean=meanArray(), std=stdArray()),
            ToTensorV2(),
        ]
    )
    testDataset = TrafficSignDataset(TRAFFIC_SIGN_DATASET_PATH / "traffic_Data" / "TEST", transforms=testTransforms)
    testDataloader = DataLoader(testDataset, batch_size=32, shuffle=False, num_workers=1)

    images, labels = None, None
    dataloaderIter = iter(testDataloader)

    while batchNumber > 0:
        batchNumber -= 1
        images, labels = next(dataloaderIter)

    images = images.to(device)  # type: ignore

    with torch.no_grad():
        outputs = inferenceModel(images)
        predictions = outputs.argmax(dim=1)

    images_cpu = images.cpu()
    preds_cpu = predictions.cpu()

    fig = plt.figure(figsize=(16, 8))

    for i in range(8):
        ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])

        img = transposeToNumpyImage(images_cpu[i])
        mean = np.array(meanArray())
        std = np.array(stdArray())
        img = img * std + mean
        img = np.clip(img, 0, 1)

        ax.imshow(img)

        pred_class_id = preds_cpu[i].item()
        pred_class_name = trainingLabels[pred_class_id].label

        # Подписываем картинку
        ax.set_title(f"Prediction:\n{pred_class_name}", fontsize=10, color="blue")

    plt.tight_layout()
    plt.show()
