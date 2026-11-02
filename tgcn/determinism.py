"""Helpers for bitwise-reproducible runs (same hardware, software and thread count).

Call reexec_with_hashseed() before importing torch, and enable_determinism() before building a model.
"""
import hashlib
import os
import random
import sys


def reexec_with_hashseed():
    if os.environ.get("PYTHONHASHSEED") != "0":
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable] + sys.argv)


def enable_determinism(seed, threads):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as e:
        print(f"[determinism] set_num_interop_threads skipped: {e}", flush=True)


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fingerprint_model(model):
    h = hashlib.sha256()
    for name, t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def fingerprint_array(*arrays):
    import numpy as np

    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def env_report():
    import torch

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "MKL_CBWR": os.environ.get("MKL_CBWR"),
        "host": os.uname().nodename,
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
    }
