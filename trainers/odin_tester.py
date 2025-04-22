import wandb
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import Trainer
from models.ODIN_Detector import ODINDetector
from dataset_loader.dataset_loader import dataset_loader

def odin_test(project_name, dataset_name, backbone_ckpt_path):
    
    wandb.init(project=project_name)
    config = wandb.config
    wandb_logger = WandbLogger(project=project_name)

    # Instantiate the ODIN detector model with sweep parameters
    odin_model = ODINDetector(
        backbone_ckpt_path=backbone_ckpt_path,
        temperature=config.temperature,
        noise_magnitude=config.noise_magnitude
    )

    # Load the dataset
    _, _, OOD_train_loader, OOD_test_loader = dataset_loader(dataset_name, config, split_test=True)

    trainer = Trainer(
        devices="auto",
        accelerator="auto",
        logger=wandb_logger,
        log_every_n_steps=1,
        callbacks=[]
    )

    trainer.fit(odin_model, OOD_train_loader)

    # Run only test
    trainer.test(odin_model, OOD_test_loader)

    wandb.finish()
