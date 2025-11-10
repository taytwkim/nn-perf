# Performance Analysis of Neural Network Training

* Model training workload to experiment with performance analysis tools like Nvidia Nsight.
* Train Resnet18 and Resnet34 on CIFAR-10

## 🚀 Setup

1. Activate a venv.
```bash!
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

2. Install dependencies.

* If on CPU or MPS (Apple Silicon):
```bash!
pip install -r requirements.txt
```

* If on NVIDIA, prefer CUDA wheels:
```bash!
pip install torch torchvision numpy pillow --index-url https://download.pytorch.org/whl/cu121
```

3. Train model (single node).

* For CPU or MPS:
```bash!
python3 train.py --model resnet18 --epochs 5 --batch-size 128
```

* On NVIDIA GPUs, enable mixed precision (amp) for speed:
```
python3 train.py --model resnet18 --epochs 20 --batch-size 256 --amp --workers 4
```

## 🚂 Arguments
```bash!
python3 train.py [--epochs <INT>] [--batch-size <INT>] [--lr <FLOAT>] 
                 [--data <PATH>] [--out-dir <DIR>]
                 [--workers <INT>] [--amp] [--label-smoothing <FLOAT>]
                 [--warmup <INT>] [--seed <INT>] [--resume <PATH>]
                 [--save-every <INT>] [--model <STRING>]
```

  * `--epochs`: Number of full passes over the training set.
  * `--batch-size`: Per-GPU/per-process mini-batch size. Global = batch_size × world_size.
  * `--lr`: Base learning rate per GPU.
  * `--data`: CIFAR-10 data directory; rank 0 downloads here if missing.
  * `--out-dir`: Output directory for checkpoints/final weights (rank 0 only).
  * `--workers`: DataLoader workers per process. CPU: 0–2; GPU: 4–8 typical.
  * `--amp`: Enable mixed precision on CUDA (ignored on CPU/MPS).
  * `--label-smoothing`: Label smoothing for cross-entropy (e.g., 0.1).
  * `--warmup`: Optimizer steps (not epochs) of linear warmup before cosine decay.
  * `--seed`: Base RNG seed (offset by rank).
  * `--resume`: Resume from checkpoint produced by this script.
  * `--save-every`: Also save a snapshot every N epochs (0 disables).
  * `--model`: specify model, either `resnet18` or `resnet34`.

## 📁 Directory
```
resnet18-cifar10/
├─ train.py
├─ data/
├─ artifacts/
└─ requirements.txt
```

* `data/` [Not tracked by git]
    * `cifar-10-python.tar.gz`: the original compressed dataset that `torchvision` downloads.
    * `cifar-10-batches-py/`: extracted from the tarball. This is what `torchvision.datasets.CIFAR10` actually reads.
        * `data_batch_1` … `data_batch_5`: 5 training batches - 10,000 images each.
        * `test_batch`: 10,000 test samples.

* `artifacts/` [Empty directory tracked by git]
    * `*_final_weights.pt` (weights only)
        * Use for inference or fine-tuning from scratch LR.
        * Contains just `model.state_dict()`.
    * `*_best.pt` (full checkpoint)
        * Use to resume training with identical optimizer dynamics.
        * Contains model weights, optimizer state (e.g., momentum), scheduler state (e.g., where you are on the LR curve), plus metadata like epoch and best accuracy.
