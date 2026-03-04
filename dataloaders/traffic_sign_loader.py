from pathlib import Path, PurePath
from typing import Optional

from helpers.dataset_wrapper import DataLabel, DatasetWrapper, ImageSample
from helpers.helpers import DATASET_DIR

TRAFFIC_SIGN_DATASET_PATH = DATASET_DIR / "traffic_sign_dataset"
TRAFFIC_SIGN_SAVE_MODEL_DIR = Path.cwd() / ".saved"
TRAFFIC_SIGN_SAVE_MODEL_FILENAME = "traffic_sign_model.pth"


class TrafficSignLabel(DataLabel):
    def __init__(self, label: str, classId: Optional[int] = None):
        super().__init__(label, classId)


class TrafficSignSample(ImageSample):
    def __init__(self, imagePath: PurePath, classId: Optional[int] = None):
        super().__init__(imagePath, classId)


class TrafficSignDataset(DatasetWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def processDataDir(self):
        for itemPath in self.dataDir.iterdir():
            if self.isValidImageFile(itemPath, valid_extensions=self.validExtensions):
                self.addSample(itemPath)
            else:
                for imagePath in itemPath.iterdir():
                    if self.isValidImageFile(imagePath, valid_extensions=self.validExtensions):
                        classId = int(itemPath.name)

                        if classId not in self.skipLabels:
                            self.addSample(imagePath, classId)
