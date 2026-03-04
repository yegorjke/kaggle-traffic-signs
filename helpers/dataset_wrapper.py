import csv
from pathlib import Path, PurePath
from typing import Dict, List, Optional, Set, Tuple

import albumentations as A
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from helpers.helpers import extractTransforms


class DataLabel:
    def __init__(self, label: str, classId: Optional[int] = None):
        self.label = label
        self.classId = classId

    def __iter__(self):
        yield self.label
        yield self.classId

    def __repr__(self):
        return f'{__class__.__name__}(classId={self.classId}, label="{self.label})"'  # type: ignore


class ImageSample:
    def __init__(self, imagePath: PurePath, classId: Optional[int] = None):
        self.imagePath = imagePath
        self.classId = classId

    def __iter__(self):
        yield self.imagePath
        yield self.classId

    def __repr__(self):
        return f'{__class__.__name__}(classId={self.classId}, imagePath="{self.imagePath})"'  # type: ignore


class DatasetWrapper(Dataset):
    def __init__(
        self,
        dataPath: Path,
        labelsPath: Optional[Path] = None,
        skipLabels: Optional[Set[int]] = None,
        validExtenstions: Optional[Set[str]] = None,
        transforms=None,
        processDataDirFn=None,
        processLabelsFileFn=None,
    ):
        self.dataDir = dataPath
        self.labelsPath = labelsPath
        self.transforms = transforms
        self.skipLabels = set(skipLabels) if skipLabels is not None else set()
        self.validExtensions = validExtenstions

        # process the labels file
        self._labels: Dict[int, DataLabel] = {}
        if processLabelsFileFn is None:
            self._labels = getLabels(labelsPath=self.labelsPath, skipLabels=skipLabels, hasHeader=True)
        else:
            self._labels = processLabelsFileFn(self, labelsPath=self.labelsPath, skipLabels=skipLabels, hasHeader=True)

        # process samples
        self._samples: List[ImageSample] = []
        if processDataDirFn is None:
            self.processDataDir()
        else:
            processDataDirFn(self)

    def processDataDir(self):
        raise NotImplementedError(f"Method {__class__.__name__}.processDataDir() is not implemented!")

    def addSample(self, imagePath, classId: Optional[int] = None):
        self._samples.append(ImageSample(imagePath, classId))

    def setTransforms(self, transforms):
        self.transforms = transforms

    @staticmethod
    def isValidImageFile(imagePath: Path, valid_extensions=None):
        result = imagePath.is_file()

        if valid_extensions is not None:
            return result and imagePath.suffix.lower() in valid_extensions

        return result

    def getLabel(self, classId: int):
        return self._labels[classId]

    @property
    def labels(self) -> Dict[int, DataLabel]:
        return self._labels

    @property
    def samples(self) -> List[ImageSample]:
        return self._samples

    @classmethod
    def getDataloader(
        cls,
        dataPath: Path,
        labelPath: Path,
        batchSize: int = 4,
        numWorkers: int = 1,
        split: Optional[float] = None,
        skipLabels: Optional[Set[int]] = None,
        validExtensions: Optional[Set[str]] = None,
        transforms=None,
        validationTransforms=None,
        processDataDirFn=None,
        processLabelsFileFn=None,
    ) -> Tuple[DataLoader[Dataset], DataLoader[Dataset]] | DataLoader[Dataset]:
        if skipLabels is None:
            skipLabels = set()

        if validExtensions is None:
            validExtensions = set()

        dataset = cls(
            dataPath,
            labelPath,
            skipLabels=skipLabels,
            validExtenstions=validExtensions,
            processDataDirFn=processDataDirFn,
            processLabelsFileFn=processLabelsFileFn,
        )

        if split is not None and (0 < split < 1):
            trainSize = int(len(dataset) * split)
            validationSize = len(dataset) - trainSize
            trainSplit, validationSplit = random_split(dataset, [trainSize, validationSize])

            trainDataset = TransformSubset(trainSplit, transforms=transforms)

            if validationTransforms is None and transforms is not None:
                validationTransforms = extractTransforms(transforms, [A.Resize, A.Normalize, A.ToTensorV2])
                # TODO: add assert

            validationDataset = TransformSubset(validationSplit, transforms=validationTransforms)

            trainDataloader = DataLoader(
                trainDataset,
                batch_size=batchSize,
                num_workers=numWorkers,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
            )
            validationDataloader = DataLoader(
                validationDataset,
                batch_size=batchSize,
                num_workers=numWorkers,
                shuffle=False,
                drop_last=True,
                pin_memory=True,
            )

            return trainDataloader, validationDataloader
        else:
            dataset.setTransforms(transforms)
            return DataLoader(
                dataset,
                batch_size=batchSize,
                num_workers=numWorkers,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
            )

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        imagePath, classId = self._samples[index]

        image = Image.open(imagePath).convert("RGB")
        image = np.array(image)  # type: ignore

        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]
        # else:
        #     image = transposeToTensor(image)

        if classId is None:
            classId = -1

        return image, classId

    def __repr__(self):
        return f"{__class__.__name__}(labelsPath={self.labelsPath},dataDir={self.dataDir})"  # type:ignore


class TransformSubset(DatasetWrapper):
    def __init__(self, subset: Subset, transforms=None):
        self.subset = subset
        self.transforms = transforms

    def __getitem__(self, index):
        image, classId = self.subset[index]

        if self.transforms is not None:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        return image, classId

    def __len__(self):
        return len(self.subset)

    def __getattr__(self, name):
        return getattr(self.subset.dataset, name)


def getLabels(
    dataset: Optional[DatasetWrapper] = None,
    labelsPath: Optional[Path] = None,
    skipLabels: Optional[Set[int]] = None,
    hasHeader: bool = False,
) -> Dict[int, DataLabel]:
    if dataset is not None:
        return dataset.labels
    elif labelsPath is not None:
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
                        labels[classId] = DataLabel(rowLabel, classId)

        return labels
    else:
        raise Exception("Provide dataset object or labelsPath!")


def getLabelsNumber(
    dataset: Optional[DatasetWrapper] = None, labelsPath: Optional[Path] = None, skipLabels: Optional[Set[int]] = None
):
    if dataset is not None:
        return len(dataset.labels)
    elif labelsPath is not None:
        return len(getLabels(labelsPath=labelsPath, skipLabels=skipLabels, hasHeader=True))
    else:
        raise Exception("Provide dataset object or labelsPath!")
