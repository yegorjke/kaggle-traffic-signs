import logging
import os
import random
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torchvision.utils import make_grid

DATASET_DIR = Path.home() / "Datasets"


def meanArray():
    return [0.485, 0.456, 0.406]


def stdArray():
    return [0.229, 0.224, 0.225]


def meanTensor():
    return torch.tensor(meanArray()).view(1, 3, 1, 1)


def stdTensor():
    return torch.tensor(stdArray()).view(1, 3, 1, 1)


def getDevice():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def saveModel(model, savePath):
    torch.save(model.state_dict(), savePath)


def exportOnnxModel(model, chw: Tuple[int, int, int], outputOnnxFile: Path):
    c, h, w = chw

    dummyInput = torch.randn(1, c, w, h)

    model.eval()

    torch.onnx.export(
        model,
        (dummyInput,),
        outputOnnxFile,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["input_image"],
        output_names=["predictions"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "predictions": {0: "batch_size"},
        },
    )


def seed(s: int = 42):
    random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transposeToNumpyImage(tensor):
    return tensor.numpy().transpose((1, 2, 0))


def transposeToTensor(image):
    return torch.from_numpy(image.transpose((2, 0, 1))).float() / 255


def imshow(imageTensor, title: Optional[str] = None):
    image = imageTensor.numpy().transpose((1, 2, 0))

    mean = np.array(meanArray())
    std = np.array(stdArray())

    image = std * image + mean
    image = np.clip(image, 0, 1)

    plt.figure(figsize=(15, 5))
    plt.imshow(image)

    if title is not None:
        plt.title(title)

    plt.axis("off")
    plt.show()


def addImageGridToTensorboard(dataloader, writer, figureName: Optional[str] = None):
    images, _ = next(iter(dataloader))
    images = images * meanTensor() + meanTensor()
    images = torch.clamp(images, 0, 1)
    grid = make_grid(images)

    if figureName is None:
        figureName = "Augmented Batch"

    writer.add_image(figureName, grid, global_step=0)


def writeConfusionMatrix(writer, currentEpoch, labels, predictions, figureName: Optional[str] = None):
    cm = confusion_matrix(labels, predictions)

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=False, cmap="Blues", fmt="g")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - Epoch {currentEpoch + 1}")

    if figureName is None:
        figureName = "Confusion Matrix"

    writer.add_figure(figureName, fig, currentEpoch)


def setupLogger(name: str, logfile: Optional[Path] = None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(logfile) if logfile is not None else None
    consoleHandler = logging.StreamHandler(sys.stdout)

    formatter = logging.Formatter("[%(name)s][%(levelname)s] - %(asctime)s - %(message)s")

    if fileHandler is not None:
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)

    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    return logger
