import os
import errno

import wandb


def _resolve_wandb_dir(config=None):
    if config is None:
        config = {}

    wandb_dir = (
        config.get("wandb_dir")
        or os.environ.get("RSG_WANDB_DIR")
        or os.environ.get("WANDB_DIR")
    )
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)
    return wandb_dir


def init_wandb_run(*, project, config=None, job_type=None, **kwargs):
    settings = kwargs.pop("settings", None)
    wandb_dir = kwargs.pop("dir", None) or _resolve_wandb_dir(config)

    if wandb_dir is not None:
        kwargs["dir"] = wandb_dir
        os.environ["WANDB_DIR"] = wandb_dir

    if settings is None and os.name == "nt":
        settings = wandb.Settings(init_timeout=120, x_service_wait=120)

    # Avoid extra local writes unless explicitly enabled by the caller.
    kwargs.setdefault("save_code", False)

    try:
        return wandb.init(
            project=project,
            config=config,
            job_type=job_type,
            settings=settings,
            **kwargs,
        )
    except OSError as exc:
        if exc.errno == errno.ENOSPC:
            print(
                "W&B initialization ran out of disk space. "
                "Retrying with W&B disabled for this run."
            )
            return wandb.init(
                project=project,
                config=config,
                job_type=job_type,
                settings=settings,
                mode="disabled",
                **kwargs,
            )
        raise
