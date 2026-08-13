import argparse
import copy
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass

from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, AutoModel, DefaultDataCollator, get_linear_schedule_with_warmup

MODEL_NAME = "google/vit-base-patch16-224-in21k"

BATCH_SIZE = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
EPOCHS = 30
WARMUP_RATIO = 0.1

RANK = 64
ROUTING_TEMPERATURE = 0.1
ADAPTER_SCALE = 4.0
LAMBDA_ROUTE = 1.0

NUM_WORKERS = min(8, os.cpu_count() or 1)
VALID_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(BASE_DIR, "data")
OUTPUT_ROOT = os.path.join(BASE_DIR, "output")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    folder_candidates: tuple
    domain_names: tuple
    domain_aliases: dict
    class_names: tuple = None


DATASETS = {
    "pacs": DatasetSpec(
        "PACS",
        ("PACS", "pacs"),
        ("art_painting", "cartoon", "photo", "sketch"),
        {
            "art_painting": 0, "art painting": 0, "artpainting": 0,
            "cartoon": 1, "photo": 2, "sketch": 3,
        },
        ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person"),
    ),
    "vlcs": DatasetSpec(
        "VLCS",
        ("VLCS", "vlcs"),
        ("Caltech101", "LabelMe", "SUN09", "VOC2007"),
        {
            "caltech101": 0, "caltech": 0,
            "labelme": 1, "sun09": 2, "sun": 2,
            "voc2007": 3, "voc": 3,
        },
        ("bird", "car", "chair", "dog", "person"),
    ),
    "officehome": DatasetSpec(
        "OfficeHome",
        ("OfficeHome", "Office-Home", "officehome", "office_home"),
        ("Product", "Clipart", "Art", "Real World"),
        {
            "product": 0, "clipart": 1, "art": 2,
            "real world": 3, "real_world": 3, "realworld": 3, "real": 3,
        },
    ),
    "digitsdg": DatasetSpec(
        "DigitsDG",
        ("DigitsDG", "Digits-DG", "digitsdg", "digits_dg", "digits"),
        ("MNIST", "MNIST-M", "SVHN", "SYN"),
        {
            "mnist": 0,
            "mnist_m": 1, "mnist-m": 1, "mnistm": 1,
            "svhn": 2,
            "syn": 3, "synth": 3, "synthetic": 3,
        },
        tuple(str(i) for i in range(10)),
    ),
    "nicopp": DatasetSpec(
        "NICO++",
        ("NICO++", "NICOpp", "NICO", "nico++", "nicopp", "nico"),
        ("Autumn", "Dim", "Grass", "Outdoor", "Rock", "Water"),
        {
            "autumn": 0, "dim": 1, "grass": 2,
            "outdoor": 3, "rock": 4, "water": 5,
        },
    ),
}


def normalize_name(value):
    return (
        value.strip().lower()
        .replace("-", "")
        .replace("_", "")
        .replace(" ", "")
        .replace("+", "")
    )


DATASET_ALIASES = {
    normalize_name("PACS"): "pacs",
    normalize_name("VLCS"): "vlcs",
    normalize_name("OfficeHome"): "officehome",
    normalize_name("Office-Home"): "officehome",
    normalize_name("DigitsDG"): "digitsdg",
    normalize_name("Digits-DG"): "digitsdg",
    normalize_name("NICO++"): "nicopp",
    normalize_name("NICOpp"): "nicopp",
    normalize_name("NICO"): "nicopp",
}


def resolve_dataset(dataset_arg):
    key = DATASET_ALIASES.get(normalize_name(dataset_arg))
    if key is None:
        supported = ", ".join(spec.name for spec in DATASETS.values())
        raise ValueError(f"Unknown dataset '{dataset_arg}'. Supported: {supported}")
    return DATASETS[key]


def resolve_dataset_dir(spec):
    for folder in spec.folder_candidates:
        path = os.path.join(DATA_ROOT, folder)
        if os.path.isdir(path):
            return path

    raise FileNotFoundError(
        f"Dataset folder for {spec.name} was not found under {DATA_ROOT}.\n"
        f"Expected for example: {os.path.join(DATA_ROOT, spec.folder_candidates[0])}"
    )


def validate_split_dirs(dataset_dir):
    split_dirs = {
        split: os.path.join(dataset_dir, split)
        for split in ("train", "val", "test")
    }

    missing = [split for split, path in split_dirs.items() if not os.path.isdir(path)]
    if missing:
        raise FileNotFoundError(
            f"Missing split folder(s): {', '.join(missing)}\n"
            f"Expected:\n"
            f"  {dataset_dir}/train/<domain>/<class>/...\n"
            f"  {dataset_dir}/val/<domain>/<class>/...\n"
            f"  {dataset_dir}/test/<domain>/<class>/..."
        )

    return split_dirs


def domain_id_from_folder(folder_name, spec):
    raw = folder_name.strip().lower()
    candidates = {
        raw,
        raw.replace("-", "_"),
        raw.replace("_", " "),
        raw.replace("-", ""),
        raw.replace("_", ""),
        raw.replace(" ", ""),
    }

    for candidate in candidates:
        if candidate in spec.domain_aliases:
            return spec.domain_aliases[candidate]

    raise ValueError(
        f"Unknown domain folder '{folder_name}' for {spec.name}. "
        f"Expected domains: {', '.join(spec.domain_names)}"
    )


def parse_numeric_prefix(folder_name):
    if folder_name.isdigit():
        return int(folder_name), folder_name

    if "_" in folder_name:
        prefix, suffix = folder_name.split("_", 1)
        if prefix.isdigit():
            return int(prefix), suffix

    return None, folder_name


def build_class_mapping(train_dir, spec):
    # PACS, VLCS and Digits-DG use the mappings from the supplied dataset code.
    if spec.class_names is not None:
        return {name.lower(): idx for idx, name in enumerate(spec.class_names)}

    observed = {}

    # Office-Home and NICO++ are discovered from the training split.
    for domain_folder in sorted(os.listdir(train_dir)):
        domain_path = os.path.join(train_dir, domain_folder)
        if not os.path.isdir(domain_path):
            continue

        for class_folder in sorted(os.listdir(domain_path)):
            class_path = os.path.join(domain_path, class_folder)
            if not os.path.isdir(class_path):
                continue

            numeric_id, class_name = parse_numeric_prefix(class_folder)
            key = class_name.lower()

            if numeric_id is not None:
                observed[key] = numeric_id
            elif key not in observed:
                observed[key] = None

    if not observed:
        raise ValueError(f"No class folders found in {train_dir}")

    if all(value is not None for value in observed.values()):
        ordered = sorted(observed.items(), key=lambda item: item[1])
    else:
        ordered = [(name, None) for name in sorted(observed)]

    return {name: idx for idx, (name, _) in enumerate(ordered)}


def class_id_from_folder(folder_name, spec, class_to_idx):
    numeric_id, class_name = parse_numeric_prefix(folder_name)

    if spec.name == "DigitsDG":
        key = str(numeric_id) if numeric_id is not None else class_name.lower()
    else:
        key = class_name.lower()

    if key in class_to_idx:
        return class_to_idx[key]

    # PACS/VLCS dataset copies can also contain folders like 000_dog.
    if numeric_id is not None and 0 <= numeric_id < len(class_to_idx):
        return numeric_id

    raise ValueError(
        f"Unknown class folder '{folder_name}' for {spec.name}. "
        f"Class key '{key}' is not in the training class mapping."
    )


class MultiDomainImageDataset(Dataset):
    def __init__(self, split_dir, processor, spec, class_to_idx):
        self.processor = processor
        self.samples = []

        for domain_folder in sorted(os.listdir(split_dir)):
            domain_path = os.path.join(split_dir, domain_folder)
            if not os.path.isdir(domain_path):
                continue

            domain_id = domain_id_from_folder(domain_folder, spec)

            for class_folder in sorted(os.listdir(domain_path)):
                class_path = os.path.join(domain_path, class_folder)
                if not os.path.isdir(class_path):
                    continue

                label_id = class_id_from_folder(class_folder, spec, class_to_idx)

                for file_name in sorted(os.listdir(class_path)):
                    if file_name.lower().endswith(VALID_EXTENSIONS):
                        self.samples.append(
                            (os.path.join(class_path, file_name), label_id, domain_id)
                        )

        if not self.samples:
            raise ValueError(f"No images found in {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label_id, domain_id = self.samples[idx]

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            pixel_values = self.processor(
                images=image,
                return_tensors="pt",
            )["pixel_values"].squeeze(0)

        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label_id, dtype=torch.long),
            "domain_labels": torch.tensor(domain_id, dtype=torch.long),
        }


class SRTALinear(nn.Module):
    def __init__(
        self,
        base_linear,
        num_domains,
        rank=RANK,
        scale=ADAPTER_SCALE,
        routing_temperature=ROUTING_TEMPERATURE,
    ):
        super().__init__()

        self.base = copy.deepcopy(base_linear)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

        self.scale = scale
        self.routing_temperature = routing_temperature

        self.V = nn.Parameter(torch.empty(base_linear.in_features, rank))
        self.C = nn.Parameter(torch.empty(rank, num_domains))
        self.G = nn.Parameter(torch.empty(num_domains, rank, rank))
        self.U = nn.Parameter(torch.empty(rank, base_linear.out_features))

        nn.init.kaiming_uniform_(self.V, a=math.sqrt(5))
        nn.init.normal_(self.C, std=0.02)
        nn.init.xavier_uniform_(self.G)
        nn.init.zeros_(self.U)

        self.routing_logits = None
        self.routing_probabilities = None

    def forward(self, x):
        base_output = self.base(x)

        z = x @ self.V
        z_route = z[:, 1:, :].mean(dim=1) if z.dim() == 3 else z

        routing_logits = (z_route @ self.C) / self.routing_temperature
        alpha = F.softmax(routing_logits, dim=-1)

        self.routing_logits = routing_logits
        self.routing_probabilities = alpha

        sample_core = torch.einsum("bd,drk->brk", alpha, self.G)
        adapted = torch.einsum("btr,brk->btk", z, sample_core)
        delta = adapted @ self.U

        return base_output + self.scale * delta


class SRTAViT(nn.Module):
    def __init__(self, num_domains, num_classes):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        self.srta_modules = nn.ModuleList()

        for layer in self.backbone.encoder.layer:
            self._replace_linear(layer.attention.attention, "query", num_domains)
            self._replace_linear(layer.attention.attention, "value", num_domains)

        self.classifier = nn.Linear(self.backbone.config.hidden_size, num_classes)

    def _replace_linear(self, parent, child_name, num_domains):
        old_linear = getattr(parent, child_name)
        new_linear = SRTALinear(old_linear, num_domains)
        setattr(parent, child_name, new_linear)
        self.srta_modules.append(new_linear)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        class_logits = self.classifier(outputs.last_hidden_state[:, 0, :])

        all_routing_logits = torch.stack(
            [module.routing_logits for module in self.srta_modules],
            dim=0,
        )

        return class_logits, all_routing_logits


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def make_loader(dataset, shuffle):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        collate_fn=DefaultDataCollator(),
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=NUM_WORKERS > 0,
    )


def evaluate(model, loader):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    domain_correct = defaultdict(int)
    domain_total = defaultdict(int)

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            domains = batch["domain_labels"].to(DEVICE, non_blocking=True)

            logits, _ = model(pixel_values)
            loss = F.cross_entropy(logits, labels)
            predictions = logits.argmax(dim=1)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

            for domain_id in domains.unique():
                domain_id_int = int(domain_id.item())
                mask = domains == domain_id
                domain_total[domain_id_int] += int(mask.sum().item())
                domain_correct[domain_id_int] += int(
                    (predictions[mask] == labels[mask]).sum().item()
                )

    domain_metrics = {}
    for domain_id in sorted(domain_total):
        domain_metrics[domain_id] = {
            "accuracy": domain_correct[domain_id] / domain_total[domain_id],
            "num_samples": domain_total[domain_id],
        }

    return (
        total_loss / max(total_samples, 1),
        total_correct / max(total_samples, 1),
        domain_metrics,
    )


def save_routing_weights(model, test_loader, spec, output_path):
    model.eval()
    num_domains = len(spec.domain_names)

    routing_sum = {
        domain_id: torch.zeros(num_domains, dtype=torch.float64)
        for domain_id in range(num_domains)
    }
    routing_count = {domain_id: 0 for domain_id in range(num_domains)}

    with torch.no_grad():
        for batch in test_loader:
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            domains = batch["domain_labels"]

            model(pixel_values)
            alpha = model.srta_modules[-1].routing_probabilities.detach().cpu()

            for row_idx, source_domain in enumerate(domains.tolist()):
                routing_sum[source_domain] += alpha[row_idx].double()
                routing_count[source_domain] += 1

    with open(output_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["input_domain", "target_pathway", "average_routing_weight", "num_samples"]
        )

        for source_domain in range(num_domains):
            count = routing_count[source_domain]
            average_alpha = (
                routing_sum[source_domain] / count
                if count
                else torch.zeros(num_domains, dtype=torch.float64)
            )

            for target_domain in range(num_domains):
                writer.writerow(
                    [
                        spec.domain_names[source_domain],
                        spec.domain_names[target_domain],
                        f"{average_alpha[target_domain].item():.8f}",
                        count,
                    ]
                )


def train(dataset_arg):
    spec = resolve_dataset(dataset_arg)
    dataset_dir = resolve_dataset_dir(spec)
    splits = validate_split_dirs(dataset_dir)

    output_dir = os.path.join(OUTPUT_ROOT, spec.name)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    training_metrics_path = os.path.join(output_dir, "training_metrics.csv")
    test_metrics_path = os.path.join(output_dir, "test_metrics.csv")
    routing_path = os.path.join(output_dir, "routing_weights.csv")
    result_path = os.path.join(output_dir, "result.json")

    class_to_idx = build_class_mapping(splits["train"], spec)
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    train_dataset = MultiDomainImageDataset(
        splits["train"], processor, spec, class_to_idx
    )
    val_dataset = MultiDomainImageDataset(
        splits["val"], processor, spec, class_to_idx
    )
    test_dataset = MultiDomainImageDataset(
        splits["test"], processor, spec, class_to_idx
    )

    num_domains = len(spec.domain_names)
    num_classes = len(class_to_idx)

    print(f"Dataset: {spec.name}")
    print(f"Device: {DEVICE}")
    print(f"Data: {dataset_dir}")
    print(f"Train / Val / Test: {len(train_dataset)} / {len(val_dataset)} / {len(test_dataset)}")
    print(f"Domains: {num_domains} | Classes: {num_classes}")

    model = SRTAViT(num_domains, num_classes).to(DEVICE)
    trainable_params, total_params = count_parameters(model)

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")

    no_decay_terms = ("bias", "LayerNorm.weight", "layernorm.weight")
    optimizer = AdamW(
        [
            {
                "params": [
                    p for name, p in model.named_parameters()
                    if p.requires_grad and not any(term in name for term in no_decay_terms)
                ],
                "weight_decay": WEIGHT_DECAY,
            },
            {
                "params": [
                    p for name, p in model.named_parameters()
                    if p.requires_grad and any(term in name for term in no_decay_terms)
                ],
                "weight_decay": 0.0,
            },
        ],
        lr=LEARNING_RATE,
    )

    train_loader = make_loader(train_dataset, True)
    val_loader = make_loader(val_dataset, False)
    test_loader = make_loader(test_dataset, False)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP_RATIO),
        num_training_steps=total_steps,
    )

    with open(training_metrics_path, "w", newline="") as file:
        csv.writer(file).writerow(
            [
                "epoch",
                "train_total_loss",
                "train_classification_loss",
                "train_routing_loss",
                "val_loss",
                "val_accuracy",
            ]
        )

    best_val_accuracy = -1.0
    best_epoch = -1

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss_sum = 0.0
        class_loss_sum = 0.0
        route_loss_sum = 0.0
        sample_count = 0

        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            domain_labels = batch["domain_labels"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            class_logits, all_routing_logits = model(pixel_values)
            classification_loss = F.cross_entropy(class_logits, labels)

            num_adapter_layers = all_routing_logits.size(0)
            routing_loss = torch.zeros((), device=DEVICE)

            for layer_index in range(num_adapter_layers):
                depth_weight = (layer_index + 1) / num_adapter_layers
                routing_loss = routing_loss + depth_weight * F.cross_entropy(
                    all_routing_logits[layer_index],
                    domain_labels,
                )

            routing_loss = routing_loss / num_adapter_layers
            loss = classification_loss + LAMBDA_ROUTE * routing_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            batch_size = labels.size(0)
            sample_count += batch_size
            total_loss_sum += loss.item() * batch_size
            class_loss_sum += classification_loss.item() * batch_size
            route_loss_sum += routing_loss.item() * batch_size

        train_total_loss = total_loss_sum / sample_count
        train_class_loss = class_loss_sum / sample_count
        train_route_loss = route_loss_sum / sample_count

        val_loss, val_accuracy, _ = evaluate(model, val_loader)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

        with open(training_metrics_path, "a", newline="") as file:
            csv.writer(file).writerow(
                [
                    epoch,
                    f"{train_total_loss:.6f}",
                    f"{train_class_loss:.6f}",
                    f"{train_route_loss:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_accuracy:.6f}",
                ]
            )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train={train_total_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_accuracy * 100:.2f}%"
        )

    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    test_loss, test_accuracy, domain_metrics = evaluate(model, test_loader)

    with open(test_metrics_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["domain", "accuracy", "num_samples"])

        for domain_id, domain_name in enumerate(spec.domain_names):
            metrics = domain_metrics.get(
                domain_id,
                {"accuracy": 0.0, "num_samples": 0},
            )
            writer.writerow(
                [
                    domain_name,
                    f"{metrics['accuracy']:.6f}",
                    metrics["num_samples"],
                ]
            )

        writer.writerow(["Overall", f"{test_accuracy:.6f}", len(test_dataset)])

    save_routing_weights(model, test_loader, spec, routing_path)

    result = {
        "dataset": spec.name,
        "model": MODEL_NAME,
        "rank": RANK,
        "routing_temperature": ROUTING_TEMPERATURE,
        "adapter_scale": ADAPTER_SCALE,
        "lambda_route": LAMBDA_ROUTE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "num_domains": num_domains,
        "num_classes": num_classes,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "trainable_parameters": trainable_params,
        "total_parameters": total_params,
        "best_epoch": best_epoch,
        "best_val_accuracy": round(best_val_accuracy, 6),
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(test_accuracy, 6),
        "checkpoint": os.path.relpath(checkpoint_path, BASE_DIR),
    }

    with open(result_path, "w") as file:
        json.dump(result, file, indent=2)

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_accuracy * 100:.2f}%")
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    print(f"Output: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the main Self-Routed Tensor Adapter (SRTA) model."
    )
    parser.add_argument(
        "dataset",
        help="PACS, VLCS, OfficeHome, DigitsDG, or NICO++",
    )
    args = parser.parse_args()
    train(args.dataset)


if __name__ == "__main__":
    main()