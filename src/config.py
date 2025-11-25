import torch

# ==================
# USER CONFIGURATION
# ==================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_classes = 2
batch_size = 64

lr = 1e-3
epochs = 10

# =======================
# DEVELOPER CONFIGURATION
# =======================

NUM_CLASSES = num_classes
BATCH_SIZE = batch_size

# PIPELINE (TRAINING & EVALUATION)

PIPELINE_CONFIG = {
    "device": device,
    "lr": lr,
    "epochs": epochs,
    "loss_class": torch.nn.CrossEntropyLoss,
    "optimizer_class": torch.optim.AdamW,
    "optimizer_kwargs": {
        "lr": 0.001, 
        "weight_decay": 0.01
    },
    "scheduler_class": torch.optim.lr_scheduler.CosineAnnealingLR,
    "scheduler_kwargs": {
        "eta_min": 0.00001,
        "T_max": 100
    }
}

# PIPELINE_CONFIG = Pipeline_Config(
#     lr=lr,
#     epochs=epochs,
#     loss_class=torch.nn.CrossEntropyLoss,
#     optimizer_class=torch.optim.AdamW,
#     optimizer_kwargs={
#         "lr": 0.001, 
#         "weight_decay": 0.01
#     },
#     scheduler_class=torch.optim.lr_scheduler.CosineAnnealingLR,
#     scheduler_kwargs={
#         "lr_min": 0.00001,
#         "max_iter": 100
#     }
# )
