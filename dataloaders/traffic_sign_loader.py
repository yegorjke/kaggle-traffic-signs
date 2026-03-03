import csv
from pathlib import Path, PurePath
from typing import Dict, List, Optional, Set

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from helpers.helpers import DATASET_DIR, transposeToTensor

TRAFFIC_SIGN_DATASET_PATH = DATASET_DIR / "traffic_sign_dataset"
TRAFFIC_SIGN_SAVE_MODEL_DIR = Path.cwd() / ".saved"
TRAFFIC_SIGN_SAVE_MODEL_FILENAME = "traffic_sign_model.pth"


class TrafficSignLabel:
    def __init__(self, label: str, classId: Optional[int] = None):
        self.label = label
        self.classId = classId

    def __iter__(self):
        yield self.label
        yield self.classId

    def __repr__(self):
        return f'{__class__.__name__}(classId={self.classId}, label="{self.label})"'  # type: ignore


class TrafficSignSample:
    def __init__(self, imagePath: PurePath, classId: Optional[int] = None):
        self.imagePath = imagePath
        self.classId = classId

    def __iter__(self):
        yield self.imagePath
        yield self.classId

    def __repr__(self):
        return f'{__class__.__name__}(classId={self.classId}, imagePath="{self.imagePath})"'  # type: ignore


class TrafficSignDataset(Dataset):
    def __init__(
        self,
        dataPath: Path,
        labelsPath: Optional[Path] = None,
        skipLabels: Optional[Set[int]] = None,
        validExtenstions: Optional[Set[str]] = None,
        transforms=None,
    ):
        self.dataDir = dataPath
        self.labelsPath = labelsPath
        self.transforms = transforms
        self.skipLabels = set(skipLabels) if skipLabels is not None else set()
        self.validExtensions = validExtenstions

        self.__labels = self.getLabels(self.labelsPath, hasHeader=True, skipLabels=skipLabels)

        self.__samples: List[TrafficSignSample] = []
        for itemPath in self.dataDir.iterdir():
            if self.isValidImageFile(itemPath, valid_extensions=self.validExtensions):
                self.addSample(itemPath)
            else:
                for imagePath in itemPath.iterdir():
                    if self.isValidImageFile(imagePath, valid_extensions=self.validExtensions):
                        classId = int(itemPath.name)

                        if classId not in self.skipLabels:
                            self.addSample(imagePath, classId)

    def addSample(self, imagePath, classId: Optional[int] = None):
        self.__samples.append(TrafficSignSample(imagePath, classId))

    @staticmethod
    def isValidImageFile(imagePath: Path, valid_extensions=None):
        result = imagePath.is_file()

        if valid_extensions is not None:
            return result and imagePath.suffix.lower() in valid_extensions

        return result

    def getLabelObject(self, classId: int):
        return self.__labels[classId]

    @classmethod
    def getLabels(
        cls,
        labelsPath: Optional[Path] = None,
        skipLabels: Optional[Set[int]] = None,
        hasHeader: bool = False,
    ) -> Dict[int, TrafficSignLabel]:
        labels = dict()
        skipLabels = set(skipLabels) if skipLabels is not None else set()
        if labelsPath is not None:
            with open(labelsPath) as csvFile:
                reader = csv.reader(csvFile, delimiter=",")

                if hasHeader:
                    next(reader)  # to skip the first row

                for rowId, rowLabel in reader:
                    classId = int(rowId)

                    if classId not in skipLabels:
                        labels[classId] = TrafficSignLabel(rowLabel, classId)

        return labels

    @property
    def labels(self) -> Dict[int, TrafficSignLabel]:
        return self.__labels

    @property
    def samples(self) -> List[TrafficSignSample]:
        return self.__samples

    def __len__(self):
        return len(self.__samples)

    def __getitem__(self, index):
        imagePath, classId = self.__samples[index]

        image = Image.open(imagePath).convert("RGB")
        image = np.array(image)  # type: ignore

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        else:
            image = transposeToTensor(image)

        if classId is None:
            classId = -1

        return image, classId

    def __repr__(self):
        return f"{__class__.__name__}(labelsPath={self.labelsPath},dataDir={self.dataDir})"  # type:ignore
