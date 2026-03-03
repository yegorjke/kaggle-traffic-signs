import argparse
from logging import Logger
from pathlib import Path
from typing import Optional, Set

import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassPrecision, MulticlassRecall
from torchvision import models

from dataloaders.traffic_sign_loader import (
    TRAFFIC_SIGN_DATASET_PATH,
    TRAFFIC_SIGN_SAVE_MODEL_DIR,
    TRAFFIC_SIGN_SAVE_MODEL_FILENAME,
    TrafficSignDataset,
)
from helpers.helpers import (
    addImageGridToTensorboard,
    getDevice,
    meanArray,
    saveModel,
    seed,
    setupLogger,
    stdArray,
    writeConfusionMatrix,
)


def train(
    model,
    trainDataloader,
    criterion,
    optimizer,
    scheduler=None,
    epochs=5,
    validationDataloader=None,
    writer=None,
    metrics: Optional[MetricCollection] = None,
    device=None,
    logger: Optional[Logger] = None,
):
    if logger is None:
        logger = setupLogger("Train")

    if device is None:
        device = getDevice()

    if metrics is not None:
        trainMetrics = metrics.clone(prefix="Train/").to(device)
        validationMetrics = metrics.clone(prefix="Validation/").to(device)

    logger.info("*" * 80)
    logger.info("Training has been started...")
    logger.info("*" * 80)

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch}/{epochs} ...")

        model.train()
        trainLoss = 0.0

        for images, labels in trainDataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            trainLoss += loss.item()
            if metrics is not None:
                trainMetrics.update(outputs, labels)

        epochTrainLoss = trainLoss / len(trainDataloader)
        logger.info(f"Train/Loss: {epochTrainLoss:.4f}")

        if writer is not None:
            writer.add_scalar("Train/Loss", epochTrainLoss, epoch)

        if metrics is not None:
            computedTrainMetrics = trainMetrics.compute()
            for name, value in computedTrainMetrics.items():
                logger.info(f"{name}: {value:.4f}")
                if writer is not None:
                    writer.add_scalar(name, value, epoch)
            trainMetrics.reset()

        # validation
        if validationDataloader is not None:
            model.eval()
            validationLoss = 0.0

            allPredictions = []
            allLabels = []

            with torch.no_grad():
                for images, labels in validationDataloader:
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    validationLoss += loss.item()
                    if metrics is not None:
                        validationMetrics.update(outputs, labels)

                    predictions = outputs.argmax(dim=1)
                    allPredictions.extend(predictions.cpu().numpy())
                    allLabels.extend(labels.cpu().numpy())

                epochValidationLoss = validationLoss / len(validationDataloader)

            if scheduler is not None:
                scheduler.step(epochValidationLoss)

            currentLR = optimizer.param_groups[0]["lr"]

            logger.info(f"Validation/Loss: {epochValidationLoss:.4f} | Current LR: {currentLR}")
            if writer is not None:
                writer.add_scalar("Validation/Loss", epochValidationLoss, epoch)
                writer.add_scalar("LR", currentLR, epoch)

            if metrics is not None:
                computedValMetrics = validationMetrics.compute()
                for name, value in computedValMetrics.items():
                    logger.info(f"{name}: {value:.4f}")
                    if writer is not None:
                        writer.add_scalar(name, value, epoch)
                validationMetrics.reset()

            if writer is not None and (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
                writeConfusionMatrix(
                    writer,
                    epoch,
                    allLabels,
                    allPredictions,
                    "Validation/ConfussionMatrix",
                )

        logger.info("-" * 30)

    logger.info("*" * 80)
    logger.info("Training has been ended!")
    logger.info("*" * 80)


if __name__ == "__main__":
    seed(42)
    logger = setupLogger("TrafficSigns", Path.cwd() / "logs" / "traffic_signs.log")

    parser = argparse.ArgumentParser("Training Loop written in PyTorch")
    parser.add_argument("-e", "--epochs", default=5, type=int, help="number of epochs")
    parser.add_argument("--b", default=32, type=int, help="the size of the batch of images")
    parser.add_argument("-w", "--workers", default=4, type=int, help="number of workers")
    parser.add_argument("-lr", "--lr", default=0.01, type=float, help="learning rate multiplier")
    parser.add_argument(
        "-s", "--split", default=0.8, type=float, help="modifier to split train and validation datasets"
    )

    args = parser.parse_args()
    epochs = args.epochs
    lr = args.lr
    split = args.split
    batchSize = args.b
    workers = args.workers
    skipLabels: Set[int] = set()
    validExtensions = set({".png", ".jpg", ".jpeg"})
    writer = SummaryWriter(log_dir="runs/" + TRAFFIC_SIGN_SAVE_MODEL_FILENAME)

    trainTransforms = A.Compose(
        [
            A.Resize(64, 64),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.Blur(blur_limit=3, p=0.2),
            A.Normalize(mean=meanArray(), std=stdArray()),
            ToTensorV2(),
        ]
    )

    trainDataset = TrafficSignDataset(
        TRAFFIC_SIGN_DATASET_PATH / "traffic_Data" / "DATA",
        TRAFFIC_SIGN_DATASET_PATH / "labels.csv",
        skipLabels=skipLabels,
        validExtenstions=validExtensions,
        transforms=trainTransforms,
    )

    trainSize = int(len(trainDataset) * split)
    validationSize = len(trainDataset) - trainSize
    trainSplit, validationSplit = random_split(trainDataset, [trainSize, validationSize])

    trainDataloader = DataLoader(trainSplit, batch_size=batchSize, num_workers=workers, shuffle=True, drop_last=True)
    validationDataloader = DataLoader(
        validationSplit, batch_size=batchSize, num_workers=workers, shuffle=False, drop_last=True
    )

    numLabels = len(trainDataset.labels)

    addImageGridToTensorboard(trainDataloader, writer, "Train/AugmentedBatch")
    addImageGridToTensorboard(validationDataloader, writer, "Validation/AugmentedBatch")

    device = getDevice()
    logger.debug(f"Going to learn on {device}")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # # freeze weights
    # for param in model.parameters():
    #     param.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, numLabels)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    metrics = MetricCollection(
        {
            "Accuracy": MulticlassAccuracy(num_classes=numLabels, average="macro").to(device),
            "Precision": MulticlassPrecision(num_classes=numLabels, average="macro").to(device),
            "Recall": MulticlassRecall(num_classes=numLabels, average="macro").to(device),
            "F1Score": MulticlassF1Score(num_classes=numLabels, average="macro").to(device),
        }
    )

    train(
        model,
        trainDataloader,
        criterion,
        optimizer,
        scheduler=scheduler,
        epochs=epochs,
        validationDataloader=validationDataloader,
        writer=writer,
        metrics=metrics,
        device=device,
        logger=logger,
    )

    writer.close()

    savePath = TRAFFIC_SIGN_SAVE_MODEL_DIR / ("resnet18_" + TRAFFIC_SIGN_SAVE_MODEL_FILENAME)
    saveModel(model, savePath)
    logger.debug(f"Model weights are saved in {savePath}")
