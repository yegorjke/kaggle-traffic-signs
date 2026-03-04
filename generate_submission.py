import os
import time
from pathlib import Path
from typing import Optional

import albumentations as A
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from dataloaders.traffic_sign_loader import (
    TRAFFIC_SIGN_DATASET_PATH,
    TRAFFIC_SIGN_SAVE_MODEL_DIR,
    TRAFFIC_SIGN_SAVE_MODEL_FILENAME,
    TrafficSignDataset,
)
from helpers.dataset_wrapper import getLabels
from helpers.helpers import getBaseImageTransforms, getDevice


def generate_submission(model, device, dataloader, outputFile: Optional[Path] = None):
    model.eval()

    allPredictions = []

    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            output = model(images)
            predictions = output.argmax(dim=1)

            allPredictions.extend(predictions.cpu().numpy())

    filenames = []
    dataset = dataloader.dataset
    for sample in dataset.samples:
        filename = os.path.basename(sample.imagePath)
        filenames.append(filename)

    submissionDataframe = pd.DataFrame({"image_id": filenames, "predicted_class": allPredictions})

    submissionDataframe.to_csv(outputFile, index=False)
    print(f'Submission "outputFile" is generated successfully: {len(allPredictions)} predictions!')


if __name__ == "__main__":
    device = getDevice()

    trainingLabels = getLabels(labelsPath=TRAFFIC_SIGN_DATASET_PATH / "labels.csv", hasHeader=True)

    modelPath = TRAFFIC_SIGN_SAVE_MODEL_DIR / ("resnet18_" + TRAFFIC_SIGN_SAVE_MODEL_FILENAME)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(trainingLabels))

    model.load_state_dict(torch.load(modelPath, weights_only=True))
    model = model.to(device)

    testTransforms = A.Compose(getBaseImageTransforms(resize=(64, 64)))
    testDataset = TrafficSignDataset(TRAFFIC_SIGN_DATASET_PATH / "traffic_Data" / "TEST", transforms=testTransforms)
    testDataloader = DataLoader(testDataset, batch_size=32, shuffle=False, num_workers=1)

    generate_submission(
        model, device, testDataloader, Path.cwd() / "submissions" / ("submission_" + str(int(time.time())) + ".csv")
    )
