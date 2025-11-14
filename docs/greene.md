# ⚡ NYU Greene Burst

This is a guide on using NYU Greene's cloud burst servers for Cloud and Machine Learning Project 1.

## Logging in

If you're on-campus using the NYU network, you can SSH directly to Greene. If not, use the [NYU VPN](https://www.nyu.edu/life/information-technology/infrastructure/network-services/vpn.html).

1. SSH to Greene's login node
```bash!
ssh <NetID>@greene.hpc.nyu.edu
```

2. From the login node, SSH again to the cloud bursting node (`log-burst`).

```bash!
ssh burst
```

A cloud bursting node is managed by Greene's scheduler, but configured to launch jobs on cloud-backed partitions (GCP spot VMs).

```
Local → Greene login
           ├─→ Greene compute nodes (on-prem partitions)
           └─→ Burst login (log-burst) → Cloud compute nodes (GCP partitions)
```

## Slurm

Slurm is a job scheduler. We ask Slurm for resources (CPU/GPU/RAM). Slurm finds a suitable compute node, runs your program there, tracks usage/quotas, and return logs when finished.

## Partitions
Paritions are specific resources (queues) on the cluster.
```
partitions = { 
    "interactive",         # for light CPU workload
    "n2c48m24",            # for heavy CPU workload
    "g2-standard-12",      # L4 
    "g2-standard-48",      # L4  
    "c12m85-a100-1",       # A100
    "c24m170-a100-2",      # A100
    "n1s8-t4-1",           # T4
}
```
* Note: all GPU hours count the same.

## Interactive vs. Batch

* *Interactive sessions*: You get a live shell. You can type commands there until you exit. This is helpful for exploration, setup, debugging, and quick tests.
* *Batch jobs*: You submit a script. Job runs unattended; we don't get a live shell, and logs go to a file. Use batch for longer experiements or multiple jobs.

## Starting Interactive Sessions

From `log-burst`, use `srun` to start interactive sessions. It might take some time before the session starts.

1. CPU-only
```bash!
srun \
--account=csci_ga_3033_085-2025fa \
--partition=interactive \
--time=00:30:00 \
--pty /bin/bash
```

2. One L4 GPU
```bash!
srun \
--account=csci_ga_3033_085-2025fa \
--partition=g2-standard-12 \
--gres=gpu:1 \
--time=01:00:00 \
--pty /bin/bash
```

3. One A100 GPU
```bash!
srun \
--account=csci_ga_3033_085-2025fa \
--partition=c12m85-a100-1 \
--gres=gpu:1 \
--time=02:00:00 \
--pty /bin/bash
```

* `--gres`: GPU requests
* `--time`: max time limit; we can exit earlier 
* `--pty`: open a live shell

Once we are done,`exit` from the session.

## File System

It is best practice to keep `$HOME` small and clean. 

Once we get a live shell from `srun`, navigate to `/scratch/$USER`. From here, we can clone our github repo and run our code.

```bash!
cd /scratch/$USER
pwd # check that we are in scratch/$USER
```

* Note: `/scratch/$USER` is mounted as a shared filesystem on the *compute nodes*, but it is not mounted the same way on `log-burst`. Don’t create/edit files under `/scratch/$USER` while on `log-burst`; do it from a compute node (after `srun`).

| Path             | Env var    | Purpose                 | Purge              | Typical use                               |
| ---------------- | ---------- | ----------------------- | ------------------ | ----------------------------------------- |
| `/home/$USER`    | `$HOME`    | Configs, small scripts  | No                 | dotfiles, SSH keys, tiny utilities        |
| `/scratch/$USER` | `$SCRATCH` | Working data (fast)     | Yes (~60 days)     | repos, datasets, logs, checkpoints        |
| `/archive/$USER` | `$ARCHIVE` | Long-term keep          | No                 | important results, datasets you must keep |

## Monitoring

If something’s stuck, find and cancel the job from another terminal. 
CF means the session is still being configured, R means your session is active.

```bash!
squeue -u $USER          # get JOBID
scancel <JOBID>          # immediately ends it
```

## Copying Files

You can't directly copy your files from burst to local. 

* Copying from GCP to Local (GCP → Greene → Local)
```bash!
# First copy from GCP to Greene
scp [optional flags] [gcp-file-path] greene-dtn:[greene-destination-path]

# Next, copy from Greene to local
scp [optional flags] greene-dtn:[file-path] [local-destination-path]
```

* Copying from Local to GCP is the other wat around (Local → Greene → GCP)