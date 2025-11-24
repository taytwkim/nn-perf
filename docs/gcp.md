# ☁️ Train in GCP

This is a guide on using GCP VMs to train.

1. Open GCP console, launch a GPU-provisioned Compute Engine (e.g., `n1-standard-4`/`T4`/`Ubuntu 22.04 LTS`). Allocate plenty of storage space (~50 GB to be safe).

2. SSH to VM, check if GPU is attached.
```bash!
lspci | grep -i "nvidia"
```

3. Check if GPU driver is available.
```bash!
nvidia-smi
```

If not, [install driver](https://cloud.google.com/compute/docs/gpus/install-drivers-gpu#linux).

4. Clone repo.
```bash!
# if necessary, install git
sudo apt-get install -y git
git clone https://github.com/taytwkim/resnet18-cifar10.git
cd resnet18-cifar10
```

5. Setup Python venv and install dependencies.
```bash!
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# install CUDA-enabled wheels (CUDA 12.1 build)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow

# quick check
python - <<'PY'
import torch
print("cuda available?:", torch.cuda.is_available())
print("cuda build:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
PY
```

6. Train model (single node).
```
python3 train.py --epochs 20 --batch-size 256 --amp --workers 4
```