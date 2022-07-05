# Standard

# PIP
from pytorch_lightning import Trainer

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

# Custom
from config import Config
from dataset import CustomDataModule
from model import CustomModule


tb_logger = TensorBoardLogger('logs/', name='classifier_finetune')
# csv_logger = CSVLogger('./', name='pretrain', version='0'),

cfg = Config()

print('main')

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    min_delta=0.00,
    patience=100,
    verbose=False,
    mode='min'
)

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath='./checkout/',
    filename='finetune-{epoch:04d}-{val_loss:.4f}',
    save_top_k=3,
    mode='min',
)

trainer = Trainer(
    gpus=[1],
    max_epochs=cfg.max_epochs,
    logger=tb_logger,
    progress_bar_refresh_rate=1,
    deterministic=True,
    precision=16,
    callbacks=[
        early_stop_callback,
        checkpoint_callback,
    ],
    num_sanity_val_steps=0,
)

data_module = CustomDataModule(
    cfg=cfg,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
)

train = data_module.train_dataloader()
val = data_module.val_dataloader()
test = data_module.test_dataloader()

model = CustomModule(
    learning_rate=cfg.learning_rate,
    criterion_name=cfg.criterion,
    optimizer_name=cfg.optimizer,
)

print('Start model fitting')
trainer.fit(model, train, val)

print('Start testing')
trainer.test(test_dataloaders=test)

print(f'Best model : {checkpoint_callback.best_model_path}')
