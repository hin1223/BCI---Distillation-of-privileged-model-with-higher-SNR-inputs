import os, sys, subprocess
import wandb
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import lightning as pl
from torchmetrics import F1Score, Recall
from sklearn.metrics import balanced_accuracy_score, f1_score
from collections import defaultdict
from pnpl.datasets import LibriBrainPhoneme, GroupedDataset
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import TQDMProgressBar

# Config

FORCE_RETRAIN_TEACHER = True
BASELINE_ONLY = True  # set False to also train teacher + distilled student

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

n_teacher = len(averaged_train_dataset)
n_student_full = len(averaged_train_dataset_student)

# Subsample baseline train data to match teacher's sample count
averaged_train_dataset_baseline = Subset(averaged_train_dataset_student, range(n_teacher))

print(f"Train samples (raw): {len(train_dataset)}")
print(f"Averaged train samples (teacher,  100x): {n_teacher}")
print(f"Averaged train samples (student,   50x): {n_student_full}")
print(f"Averaged train samples (baseline,  50x, subsampled to match teacher): {len(averaged_train_dataset_baseline)}")

# Dataloaders

num_workers = 0
batch_size = 16

train_dl_teacher = DataLoader(averaged_train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
val_dl_teacher = DataLoader(averaged_val_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers)

train_dl_student = DataLoader(averaged_train_dataset_student, batch_size = batch_size, shuffle = True, num_workers = num_workers)
val_dl_student = DataLoader(averaged_val_dataset_student, batch_size = batch_size, shuffle = False, num_workers = num_workers)

train_dl_baseline = DataLoader(averaged_train_dataset_baseline, batch_size = batch_size, shuffle = True, num_workers = num_workers)

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
        self.balanced_acc = Recall(num_classes = num_classes, average = "macro", task = "multiclass")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        bal_acc = self.balanced_acc(y_hat, y)
        self.log("train_loss", loss, prog_bar = True, on_step = True, on_epoch = True)
        self.log("train_f1_macro", f1_macro, prog_bar = True, on_step = True, on_epoch = True)
        self.log("train_balanced_acc", bal_acc, prog_bar = True, on_step = True, on_epoch = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        f1_macro = self.f1_macro(y_hat, y)
        bal_acc = self.balanced_acc(y_hat, y)
        self.log("val_loss", loss, prog_bar = True)
        self.log("val_f1_macro", f1_macro, prog_bar = True)
        self.log("val_balanced_acc", bal_acc, prog_bar = True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr = 5e-4)

if not BASELINE_ONLY:

    # Train / load teacher model (100x averaging)

    teacher_ckpt = f"{base_path}/models/phoneme_teacher.ckpt"
    os.makedirs(os.path.dirname(teacher_ckpt), exist_ok = True)

    if os.path.exists(teacher_ckpt) and FORCE_RETRAIN_TEACHER:
        os.remove(teacher_ckpt)

    if os.path.exists(teacher_ckpt):
        print("Loading existing teacher model from checkpoint...")
        teacher_model = PhonemeClassificationModel.load_from_checkpoint(teacher_ckpt)
    else:
        print("Training teacher model from scratch...")
        teacher_model = PhonemeClassificationModel(num_classes = len(train_dataset.labels_sorted))
        logger_teacher = WandbLogger(project = "bci-distillation", name = "phoneme_teacher", log_model = True)
        trainer_teacher = pl.Trainer(
            logger = logger_teacher,
            max_epochs = 10,
            devices = "auto",
            callbacks = [TQDMProgressBar(refresh_rate = 10)],
        )
        trainer_teacher.fit(teacher_model, train_dl_teacher, val_dl_teacher)
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
        self.balanced_acc = Recall(num_classes = num_classes, average = "macro", task = "multiclass")

    def forward(self, x):
        return self.student(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        with torch.no_grad():
            teacher_logits = self.teacher(x)

        student_logits = self.student(x)
        ce_loss = self.ce(student_logits, y)

        p_stu = F.log_softmax(student_logits / self.temperature, dim = 1)
        p_tea = F.softmax(teacher_logits / self.temperature, dim = 1)
        kd_loss = F.kl_div(p_stu, p_tea, reduction = "batchmean") * (self.temperature ** 2)

        loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss

        f1_macro = self.f1_macro(student_logits, y)
        bal_acc = self.balanced_acc(student_logits, y)
        self.log("train_loss", loss, prog_bar = True, on_step = True, on_epoch = True)
        self.log("train_f1_macro", f1_macro, prog_bar = True, on_step = True, on_epoch = True)
        self.log("train_balanced_acc", bal_acc, prog_bar = True, on_step = True, on_epoch = True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        student_logits = self.student(x)
        loss = self.ce(student_logits, y)
        f1_macro = self.f1_macro(student_logits, y)
        bal_acc = self.balanced_acc(student_logits, y)
        self.log("val_loss", loss, prog_bar = True)
        self.log("val_f1_macro", f1_macro, prog_bar = True)
        self.log("val_balanced_acc", bal_acc, prog_bar = True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.student.parameters(), lr = 5e-4)
    
if not BASELINE_ONLY:

    # Train student distillation model (50x averaging)

    distilled_model = DistilledPhonemeModel(
        teacher_model = teacher_model,
        temperature = 2.0,
        alpha = 0.5,
        num_classes = len(train_dataset.labels_sorted)
    )

    logger_student = WandbLogger(project = "bci-distillation", name = "phoneme_student_distilled", log_model = True)
    trainer_student = pl.Trainer(
        logger = logger_student,
        max_epochs = 15,
        devices = "auto",
        callbacks = [TQDMProgressBar(refresh_rate = 10)],
    )

    trainer_student.fit(distilled_model, train_dl_student, val_dl_student)

    student_ckpt = f"{base_path}/models/phoneme_student.ckpt"
    trainer_student.save_checkpoint(student_ckpt)

# Baseline: student trained directly with 50x averaging (no distillation, against true labels)

baseline_model = PhonemeClassificationModel(num_classes = len(train_dataset.labels_sorted))

logger_baseline = WandbLogger(project = "bci-distillation", name = "phoneme_baseline_student", log_model = True)
trainer_baseline = pl.Trainer(
    logger = logger_baseline,
    max_epochs = 15,
    devices = "auto",
    callbacks = [TQDMProgressBar(refresh_rate = 10)],
)

trainer_baseline.fit(baseline_model, train_dl_baseline, val_dl_student)

baseline_ckpt = f"{base_path}/models/phoneme_baseline.ckpt"
trainer_baseline.save_checkpoint(baseline_ckpt)

# Evaluate + visualise classwise scores

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

    overall_f1 = F1Score(task = "multiclass", average = "macro", num_classes = len(labels)).to(module.device)
    f1_macro = overall_f1(predicted, true)

    overall_bal = Recall(task = "multiclass", average = "macro", num_classes = len(labels)).to(module.device)
    balanced_acc = overall_bal(predicted, true)

    binary_f1 = F1Score(task = "binary").to(module.device)
    f1_by_class = []
    for c in range(len(labels)):
        f1_class = binary_f1(predicted == c, true == c)
        f1_by_class.append(f1_class)
    f1_by_class = torch.stack(f1_by_class)

    return f1_macro, balanced_acc, f1_by_class

# Baseline student evaluated on 50x averaged validation
baseline_f1_macro, baseline_bal_acc, baseline_f1_by_class = validate(val_dl_student, baseline_model, val_dataset.labels_sorted)
print(f"Baseline student    — F1 Macro: {baseline_f1_macro.item():.4f}  Balanced Acc: {baseline_bal_acc.item():.4f}  (50x val)")

summary = {
    "baseline/val_f1_macro_50x":     baseline_f1_macro.item(),
    "baseline/val_balanced_acc_50x": baseline_bal_acc.item(),
}

if not BASELINE_ONLY:
    # Teacher evaluated on 100x averaged validation
    teacher_f1_macro, teacher_bal_acc, teacher_f1_by_class = validate(val_dl_teacher, teacher_model, val_dataset.labels_sorted)
    print(f"Teacher model       — F1 Macro: {teacher_f1_macro.item():.4f}  Balanced Acc: {teacher_bal_acc.item():.4f}  (100x val)")

    # Distilled student evaluated on 50x averaged validation
    student_f1_macro, student_bal_acc, student_f1_by_class = validate(val_dl_student, distilled_model, val_dataset.labels_sorted)
    print(f"Student distilled   — F1 Macro: {student_f1_macro.item():.4f}  Balanced Acc: {student_bal_acc.item():.4f}  (50x val)")

    summary.update({
        "teacher/val_f1_macro_100x":             teacher_f1_macro.item(),
        "teacher/val_balanced_acc_100x":          teacher_bal_acc.item(),
        "student_distilled/val_f1_macro_50x":     student_f1_macro.item(),
        "student_distilled/val_balanced_acc_50x": student_bal_acc.item(),
    })

with wandb.init(project = "bci-distillation", name = "eval_summary", reinit = True):
    wandb.log(summary)

# Visualise per-class F1 scores
models_info = [(baseline_f1_by_class, "Baseline Student (50x avg)", "orange")]
if not BASELINE_ONLY:
    models_info = [
        (teacher_f1_by_class,  "Teacher (100x avg)",          "green"),
        (student_f1_by_class,  "Student Distilled (50x avg)", "blue"),
        (baseline_f1_by_class, "Baseline Student (50x avg)",  "orange"),
    ]

fig, axes = plt.subplots(len(models_info), 1, figsize = (20, 8 * len(models_info)))
axes = [axes] if len(models_info) == 1 else axes

for ax, (f1_vals, title, color) in zip(axes, models_info):
    ax.bar(range(len(f1_vals)), f1_vals.cpu().numpy(), color = color)
    ax.set_xticks(range(len(f1_vals)))
    ax.set_xticklabels(val_dataset.labels_sorted, rotation = 90)
    ax.set_ylabel("F1 By Phoneme")
    ax.set_title(f"Phoneme-wise F1 Scores ({title})")

plt.tight_layout()
plt.show()

print("Training and evaluation completed successfully!")