# Kaggle: Traffic Signs

## Requirements

### Install dependecies

```bash
uv venv .venv
uv pip install -r requirements.txt
```

### Download dataset

```
https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification
```

You can use another traffic sign dataset. You need simply change some code in `traffic_sign_loader.py` to match pathes for train and test of yours dataset.

## Usage

### Training

```bash
python train.py --epochs 5 --lr 0.0005 --split 0.8 --b 32 --workers 4
```

Optional params:

`--epochs` - number of epochs

`--lr` - learning rate

`--split` - train/validation split modifier

`--b` - batch size

`--workers` - number of workers

### Inference

```bash
python inference.py --b 1
```

Optional params:

`--b` optional parameter means what number of batch to process (8 images per batch)


### Generate submission

```bash
python generate_submission.py
```
