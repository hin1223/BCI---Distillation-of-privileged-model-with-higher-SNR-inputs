import os, sys, subprocess
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import lightning as pl
from torchmetrics import F1Score
from sklearn.metrics import balanced_accuracy_score, f1_score
from collections import defaultdict
from pnpl.datasets import LibriBrainPhoneme, GroupedDataset
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import TQDMProgressBar

# Reproducibility

pl.seed_everything(42)

# Metrics

def compute_metrics(y_true, y_pred, task = "phoneme"):
    if task == "speech":
        metric = balanced_accuracy_score(y_true, y_pred)
    elif task == "phoneme":
        metric = f1_score(y_true, y_pred, average = "macro")
    else:
        raise ValueError(f"Unknown task: {task}")
    return metric

# Load LibriBrainPhoneme datasets

base_path = "./libribrain"
os.makedirs(base_path, exist_ok = True)

train_dataset = LibriBrainPhoneme(
    data_path = f"{base_path}/data",
    include_run_keys = [("0", str(i), "Sherlock1", "1") for i in range(1, 10)],
    tmin = 0.0,
    tmax = 0.5
)

val_dataset = LibriBrainPhoneme(
    data_path = f"{base_path}/data",
    partition = "validation",
    tmin = 0.0,
    tmax = 0.5
)

test_dataset = LibriBrainPhoneme(
    data_path = f"{base_path}/data",
    partition = "test",
    tmin = 0.0,
    tmax = 0.5
)

averaged_train_dataset = GroupedDataset(train_dataset, grouped_samples = 100)
averaged_val_dataset = GroupedDataset(val_dataset, grouped_samples = 100)
averaged_test_dataset = GroupedDataset(test_dataset, grouped_samples = 100)

averaged_train_dataset_student = GroupedDataset(train_dataset, grouped_samples = 50)
averaged_val_dataset_student = GroupedDataset(val_dataset, grouped_samples = 50)
averaged_test_dataset_student = GroupedDataset(test_dataset, grouped_samples = 50)

print(f"Train samples: {len(train_dataset)}")
print(f"Averaged train samples: {len(averaged_train_dataset)}")

# Define base phoneme model (teacher / student architecture)

class PhonemeClassificationModel(pl.LightningModule):
    def __init__(self, num_classes = 39):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(306, 128, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 125, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.f1_macro = F1Score(num_classes = num_classes, average = "macro", task = "multiclass")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        self.log("train_loss", loss, prog_bar = True)
        self.log("train_f1_macro", f1_macro)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        self.log("val_loss", loss, prog_bar = True)
        self.log("val_f1_macro", f1_macro, prog_bar = True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = 5e-4)

# Train / load teacher model

LOG_DIR = f"{base_path}/lightning_logs"
os.makedirs(LOG_DIR, exist_ok = True)
logger = TensorBoardLogger(save_dir = LOG_DIR, name = "phoneme_teacher", default_hp_metric = True)

num_workers = 0
train_dl = DataLoader(averaged_train_dataset, batch_size = 16, shuffle = True, num_workers = num_workers)
val_dl = DataLoader(val_dataset, batch_size = 16, shuffle = False, num_workers = num_workers)

teacher_ckpt = f"{base_path}/models/phoneme_teacher.ckpt"
os.makedirs(os.path.dirname(teacher_ckpt), exist_ok = True)

if os.path.exists(teacher_ckpt):
    print("Loading existing teacher model from checkpoint...")
    teacher_model = PhonemeClassificationModel.load_from_checkpoint(teacher_ckpt)
else:
    print("Training teacher model from scratch...")
    teacher_model = PhonemeClassificationModel(num_classes = len(train_dataset.labels_sorted))
    trainer_teacher = pl.Trainer(
        logger = logger,
        max_epochs = 10,
        devices = "auto",
        callbacks = [TQDMProgressBar(refresh_rate = 10)],
    )
    trainer_teacher.fit(teacher_model, train_dl, val_dl)
    trainer_teacher.save_checkpoint(teacher_ckpt)

teacher_model.eval()
    
# Define student distillation model

class DistilledPhonemeModel(pl.LightningModule):

    def __init__(self, teacher_model, temperature = 2.0, alpha = 0.5, num_classes = 39):
        super().__init__()
        self.student = nn.Sequential(
            nn.Conv1d(306, 128, 1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 125, num_classes)
        )
        
        self.teacher = teacher_model.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        
        self.temperature = temperature
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.f1_macro = F1Score(num_classes = num_classes, average = 'macro', task = "multiclass")
    
    def forward(self, x):
        return self.student(x)
    
    def training_step(self, batch, batch_idx):
        x, y =  batch
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        
        student_logits = self.student(x)
        ce_loss = self.ce(student_logits, y)

        p_stu = F.log_softmax(student_logits / self.temperature, dim = 1)
        p_tea = F.softmax(teacher_logits / self.temperature, dim = 1)
        kd_loss = F.kl_div(p_stu, p_tea, reduction = "batchmean") * (self.temperature ** 2)

        loss = self.alpha * kd_loss + (1-self.alpha) * ce_loss

        f1_macro = self.f1_macro(student_logits, y)
        self.log("train_loss", loss, prog_bar = True)
        self.log("train_f1_macro", f1_macro)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        student_logits = self.student(x)
        loss = self.ce(student_logits, y)
        f1_macro = self.f1_macro(student_logits, y)
        self.log("val_loss", loss, prog_bar = True)
        self.log("val_f1_macro", f1_macro, prog_bar = True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.student.parameters(), lr = 5e-4)
    
# Train student distillation model

distilled_model = DistilledPhonemeModel(
    teacher_model = teacher_model,
    temperature = 2.0,
    alpha = 0.5,
    num_classes = len(train_dataset.labels_sorted) # look into this line re what labels_sorted does
)

logger_student = TensorBoardLogger(save_dir = LOG_DIR, name = "phoneme_student", default_hp_metric = True)
trainer_student = pl.Trainer(
    logger = logger_student,
    max_epochs = 15,
    devices = "auto",
    callbacks = [TQDMProgressBar(refresh_rate = 10)],
)

trainer_student.fit(distilled_model, train_dl, val_dl)

student_ckpt = f"{base_path}/models/phoneme_student.ckpt"
trainer_student.save_checkpoint(student_ckpt)

# Evaluate + visualise classwise scores

from torchmetrics import F1Score

def validate(val_loader, module, labels):
    module.eval()
    predicted, true = [], []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(module.device)
            y = y.to(module.device)
            out = module(x)
            preds = torch.argmax(out, dim = 1)
            predicted.extend(preds)
            true.extend(y)

    predicted = torch.stack(predicted)
    true = torch.stack(true)

    overall = F1Score(task = "multiclass", average = "macro", num_classes = len(labels)).to(module.device)
    f1_macro = overall(predicted, true)

    binary_f1 = F1Score(task = "binary").to(module.device)
    f1_by_class = []
    for c in range(len(labels)):
        f1_class = binary_f1(predicted == c, true == c)
        f1_by_class.append(f1_class)
    f1_by_class = torch.stack(f1_by_class)

    return f1_macro, f1_by_class

val_f1_macro, f1_by_class = validate(val_dl, distilled_model, val_dataset.labels_sorted)
print("Student distilled model F1 Macro", val_f1_macro.item())

plt.figure(figsize = (20, 8))
plt.bar(range(len(f1_by_class)), f1_by_class.cpu().numpy(), color = 'blue')
plt.xticks(range(len(f1_by_class)), val_dataset.labels_sorted, rotation = 90)
plt.ylabel("F1 By Phoneme")
plt.title("Phoneme-wise F1 Scores (Student Model)")
plt.tight_layout()
plt.show()

print("Training and evaluation completed successfully!")