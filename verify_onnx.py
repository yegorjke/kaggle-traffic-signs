import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import models

from dataloaders.traffic_sign_loader import (
    TRAFFIC_SIGN_SAVE_MODEL_DIR,
    TRAFFIC_SIGN_SAVE_MODEL_FILENAME,
    getLabelsNumber,
)
from helpers.helpers import seed

if __name__ == "__main__":
    seed(42)
    pytorchModelPath = TRAFFIC_SIGN_SAVE_MODEL_DIR / f"resnet18_{TRAFFIC_SIGN_SAVE_MODEL_FILENAME}"

    numLabels = getLabelsNumber()
    device = torch.device("cpu")

    pytorchModel = models.resnet18(weights=None)
    pytorchModel.fc = nn.Linear(pytorchModel.fc.in_features, numLabels)
    pytorchModel.load_state_dict(torch.load(pytorchModelPath, map_location=device, weights_only=True))
    pytorchModel.eval()

    # onnx
    onnxFilename = TRAFFIC_SIGN_SAVE_MODEL_FILENAME[: TRAFFIC_SIGN_SAVE_MODEL_FILENAME.index(".")]
    onnxModelPath = TRAFFIC_SIGN_SAVE_MODEL_DIR / f"resnet18_{onnxFilename}.onnx"
    session = ort.InferenceSession(onnxModelPath)

    # run test
    testInput = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        pytorchOutput = pytorchModel(testInput).numpy()

    ortInputs = {"input_image": testInput.numpy()}
    onnxOutput = session.run(None, ortInputs)[0]

    # verify
    try:
        np.testing.assert_allclose(pytorchOutput, onnxOutput, rtol=1e-03, atol=1e-05)
        print(f"PyTorch: {pytorchOutput[0][:5], {np.argmax(pytorchOutput[0])}}")
        print(f"ONNX:    {onnxOutput[0][:5]}, {np.argmax(onnxOutput[0])}")
    except AssertionError as e:
        print(f"X: {e}")
