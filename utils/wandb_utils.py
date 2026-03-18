import os

import wandb


def init_wandb_run(*, project, config=None, job_type=None, **kwargs):
    settings = kwargs.pop("settings", None)

    if settings is None and os.name == "nt":
        settings = wandb.Settings(init_timeout=120, x_service_wait=120)

    return wandb.init(
        project=project,
        config=config,
        job_type=job_type,
        settings=settings,
        **kwargs,
    )
