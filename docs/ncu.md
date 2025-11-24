# Install Nsight Compute

There are several ways to install NCU. A commonly used approach is to install the NVIDIA CUDA Toolkit, which ships with NCU, NSYS, and other GPU profiling and development utilities.

1. Check the OS/version
```bash
cat /etc/os-release
```

2. Then, download the matching Toolkit runfile. Check [Nvidia's documentation](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions) to find the right command/version. The command below downloads version 12.4 on RHEL.

```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run -O cuda.run
```

3. This creates `cuda.run`, which we use to install the Toolkit.
```bash
chmod +x cuda.run
```

4. Install the Toolkit. This creates a directory called cuda-12.4, under which CUDA tools are available.
```bash
bash ./cuda.run --silent --toolkit --override --installpath=/scratch/$USER/cuda-12.4
```
* CUDA runfile may install GPU driver along with CUDA Toolkit. We might not want to replace the system's driver (if we are on a shared machine like the school server), so use the `--toolkit` argument to install only the toolkit.
* Also, don't use sudo just in case - we don't want to touch the system files. Install into the user directory.


5. Next, wire up the shell. `~/.bashrc` is a startup script for bash. We can configure with custom variables and reload it to apply changes to the current shell.
```bash
echo 'export CUDA_HOME=/scratch/$USER/cuda-12.4' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```


6. Verify that the tools we need are there.
```
which ncu && ncu --version
which nsys && nsys --version
```