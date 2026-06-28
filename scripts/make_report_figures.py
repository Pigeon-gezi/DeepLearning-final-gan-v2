from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "report" / "assets"

LOGS = {
    "DCGAN": ROOT / "runs" / "dcgan_celeba" / "train_log.jsonl",
    "StyleGAN-lite": ROOT / "runs" / "stylegan_lite_celeba" / "train_log.jsonl",
    "StyleGAN-lite-v2": ROOT / "runs" / "stylegan_lite_v2_celeba" / "train_log.jsonl",
}

EVALS = {
    "DCGAN": ROOT / "runs" / "dcgan_celeba" / "evaluations" / "best_fid_n10000_20260628_003658.json",
    "StyleGAN-lite": ROOT
    / "runs"
    / "stylegan_lite_celeba"
    / "evaluations"
    / "best_fid_n10000_20260628_010315.json",
    "StyleGAN-lite-v2": ROOT
    / "runs"
    / "stylegan_lite_v2_celeba"
    / "evaluations"
    / "best_fid_n10000_manual.json",
}

COLORS = {
    "DCGAN": "#3568a8",
    "StyleGAN-lite": "#bf3f3f",
    "StyleGAN-lite-v2": "#2d8a5f",
}


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_eval(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_rows(rows: Iterable[dict], key: str) -> tuple[list[int], list[float]]:
    epochs: list[int] = []
    values: list[float] = []
    for row in rows:
        if key in row:
            epochs.append(int(row["epoch"]))
            values.append(float(row[key]))
    return epochs, values


def save_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_fid_curve(logs: dict[str, list[dict]]) -> None:
    plt.figure(figsize=(7.2, 4.2))
    for name, rows in logs.items():
        epochs, values = metric_rows(rows, "fid")
        plt.plot(epochs, values, marker="o", linewidth=2, label=name, color=COLORS[name])
    plt.xlabel("Epoch")
    plt.ylabel("FID (lower is better)")
    plt.title("FID during training")
    plt.grid(True, alpha=0.25)
    plt.legend()
    save_plot(ASSETS / "fid_curve.png")


def plot_losses(logs: dict[str, list[dict]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.0), sharex=False)
    for ax, (name, rows) in zip(axes, logs.items()):
        epochs = [int(row["epoch"]) for row in rows]
        loss_g = [float(row["loss_g"]) for row in rows]
        loss_d = [float(row["loss_d"]) for row in rows]
        ax.plot(epochs, loss_g, label="Generator", color="#7851a9", linewidth=1.8)
        ax.plot(epochs, loss_d, label="Discriminator", color="#d18f28", linewidth=1.8)
        ax.set_title(name)
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Epoch")
    fig.suptitle("Generator and discriminator losses", y=1.01)
    save_plot(ASSETS / "loss_curves.png")


def plot_diversity(rows: list[dict]) -> None:
    epochs, ms_ssim = metric_rows(rows, "diversity_ms_ssim")
    _, diversity = metric_rows(rows, "diversity_score")
    plt.figure(figsize=(7.2, 4.0))
    plt.plot(epochs, ms_ssim, marker="o", label="MS-SSIM (lower)", color="#bf3f3f", linewidth=2)
    plt.plot(epochs, diversity, marker="o", label="1 - MS-SSIM (higher)", color="#2d8a5f", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("StyleGAN-lite-v2 diversity metrics")
    plt.grid(True, alpha=0.25)
    plt.legend()
    save_plot(ASSETS / "v2_diversity_curve.png")


def plot_final_bars(evals: dict[str, dict]) -> None:
    names = list(evals)
    fid = [float(evals[name]["metrics"]["fid"]) for name in names]
    is_mean = [float(evals[name]["metrics"]["is_mean"]) for name in names]
    is_std = [float(evals[name]["metrics"]["is_std"]) for name in names]
    diversity = [float(evals[name]["metrics"]["diversity_score"]) for name in names]
    colors = [COLORS[name] for name in names]

    plt.figure(figsize=(6.4, 4.0))
    plt.bar(names, fid, color=colors)
    plt.ylabel("FID (lower is better)")
    plt.title("Final FID on 10,000 generated samples")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(fid):
        plt.text(idx, value + max(fid) * 0.02, f"{value:.2f}", ha="center", fontsize=9)
    save_plot(ASSETS / "final_fid_bar.png")

    plt.figure(figsize=(6.4, 4.0))
    plt.bar(names, is_mean, yerr=is_std, capsize=4, color=colors)
    plt.ylabel("Inception Score")
    plt.title("Final Inception Score on 10,000 generated samples")
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(is_mean):
        plt.text(idx, value + 0.08, f"{value:.2f}", ha="center", fontsize=9)
    save_plot(ASSETS / "final_is_bar.png")

    plt.figure(figsize=(6.4, 4.0))
    plt.bar(names, diversity, color=colors)
    plt.ylabel("Diversity score (1 - MS-SSIM)")
    plt.title("Final diversity on 512 generated samples")
    plt.xticks(rotation=15, ha="right")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(diversity):
        plt.text(idx, value + 0.025, f"{value:.3f}", ha="center", fontsize=9)
    save_plot(ASSETS / "final_diversity_bar.png")


def plot_metrics_dashboard(logs: dict[str, list[dict]], evals: dict[str, dict]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.0))
    ax = axes[0, 0]
    for name, rows in logs.items():
        epochs, values = metric_rows(rows, "fid")
        ax.plot(epochs, values, marker="o", linewidth=1.8, label=name, color=COLORS[name])
    ax.set_title("FID during training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("FID")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    names = list(evals)
    fid = [float(evals[name]["metrics"]["fid"]) for name in names]
    ax.bar(names, fid, color=[COLORS[name] for name in names])
    ax.set_title("Final FID")
    ax.set_ylabel("FID")
    ax.tick_params(axis="x", labelrotation=15)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1, 0]
    rows = logs["StyleGAN-lite-v2"]
    epochs, values = metric_rows(rows, "diversity_score")
    ax.plot(epochs, values, marker="o", color=COLORS["StyleGAN-lite-v2"], linewidth=1.8)
    ax.set_title("StyleGAN-lite-v2 diversity")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("1 - MS-SSIM")
    ax.grid(True, alpha=0.25)

    ax = axes[1, 1]
    rows = logs["StyleGAN-lite-v2"]
    epochs = [int(row["epoch"]) for row in rows]
    ax.plot(epochs, [float(row["loss_g"]) for row in rows], label="G", color="#7851a9")
    ax.plot(epochs, [float(row["loss_d"]) for row in rows], label="D", color="#d18f28")
    ax.set_title("StyleGAN-lite-v2 losses")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(True, alpha=0.25)
    ax.legend()
    save_plot(ASSETS / "metrics_dashboard.png")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "msyh.ttc", "simhei.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def add_title(image: Image.Image, title: str) -> Image.Image:
    font = load_font(34)
    pad = 18
    title_h = 56
    out = Image.new("RGB", (image.width, image.height + title_h), "white")
    out.paste(image.convert("RGB"), (0, title_h))
    draw = ImageDraw.Draw(out)
    draw.text((pad, 10), title, fill=(20, 20, 20), font=font)
    return out


def resize_width(image: Image.Image, width: int) -> Image.Image:
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def concat_vertical(items: list[Image.Image], gap: int = 18) -> Image.Image:
    width = max(item.width for item in items)
    height = sum(item.height for item in items) + gap * (len(items) - 1)
    out = Image.new("RGB", (width, height), "white")
    y = 0
    for item in items:
        out.paste(item.convert("RGB"), ((width - item.width) // 2, y))
        y += item.height + gap
    return out


def concat_horizontal(items: list[Image.Image], gap: int = 18) -> Image.Image:
    height = max(item.height for item in items)
    width = sum(item.width for item in items) + gap * (len(items) - 1)
    out = Image.new("RGB", (width, height), "white")
    x = 0
    for item in items:
        out.paste(item.convert("RGB"), (x, (height - item.height) // 2))
        x += item.width + gap
    return out


def make_image_composites() -> None:
    sample_paths = [
        ("DCGAN", ROOT / "runs" / "dcgan_celeba" / "best_samples.png"),
        ("StyleGAN-lite", ROOT / "runs" / "stylegan_lite_celeba" / "samples" / "epoch_0100.png"),
        ("StyleGAN-lite-v2", ROOT / "runs" / "stylegan_lite_v2_celeba" / "samples" / "epoch_0150.png"),
    ]
    sample_items = []
    for title, path in sample_paths:
        image = resize_width(Image.open(path), 900)
        sample_items.append(add_title(image, title))
    concat_vertical(sample_items).save(ASSETS / "sample_grid_comparison.png")

    interp_paths = [
        ("DCGAN z", ROOT / "runs" / "dcgan_celeba" / "interpolation_z_00.png"),
        ("StyleGAN-lite-v2 z", ROOT / "runs" / "stylegan_lite_v2_celeba" / "interpolations" / "interpolation_z_00.png"),
        ("StyleGAN-lite-v2 w", ROOT / "runs" / "stylegan_lite_v2_celeba" / "interpolations" / "interpolation_w_00.png"),
    ]
    interp_items = []
    for title, path in interp_paths:
        image = resize_width(Image.open(path), 900)
        interp_items.append(add_title(image, title))
    concat_vertical(interp_items).save(ASSETS / "interpolation_comparison.png")

    # Copy key single images into report/assets with stable names.
    copies = {
        "dcgan_samples.png": ROOT / "runs" / "dcgan_celeba" / "best_samples.png",
        "stylegan_lite_failure_epoch_0020.png": ROOT / "runs" / "stylegan_lite_celeba" / "samples" / "epoch_0020.png",
        "stylegan_lite_failure_epoch_0100.png": ROOT / "runs" / "stylegan_lite_celeba" / "samples" / "epoch_0100.png",
        "stylegan_lite_v2_samples.png": ROOT / "runs" / "stylegan_lite_v2_celeba" / "samples" / "epoch_0150.png",
        "dcgan_interp_z.png": ROOT / "runs" / "dcgan_celeba" / "interpolation_z_00.png",
        "stylegan_v2_interp_z.png": ROOT
        / "runs"
        / "stylegan_lite_v2_celeba"
        / "interpolations"
        / "interpolation_z_00.png",
        "stylegan_v2_interp_w.png": ROOT
        / "runs"
        / "stylegan_lite_v2_celeba"
        / "interpolations"
        / "interpolation_w_00.png",
    }
    for target, source in copies.items():
        Image.open(source).save(ASSETS / target)


def node_color(label: str) -> tuple[str, str]:
    label_lower = label.lower()
    if any(token in label_lower for token in ("celeba", "real", "preprocess", "image")):
        return "#e8f3f1", "#2f6f63"
    if any(token in label_lower for token in ("train", "generator", "discriminator", "styledconv", "mapping", "dcgan", "stylegan")):
        return "#eef1fb", "#4a5f9e"
    if any(token in label_lower for token in ("fid", "is", "ms-ssim", "evaluation", "statistics", "loss")):
        return "#fff4df", "#a46c12"
    if any(token in label_lower for token in ("report", "json", "samples", "interpolation", "torgb")):
        return "#f4ecf7", "#7b4b8d"
    return "#f5f7fa", "#2d4059"


def draw_flow(
    path: Path,
    boxes: list[tuple[str, float, float, float, float]],
    arrows: list[tuple[int, int]],
    title: str,
    x_max: float = 10,
) -> None:
    fig, ax = plt.subplots(figsize=(9.0 * x_max / 10, 3.8))
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title(title, fontsize=15, pad=12, fontweight="bold", color="#263238")
    for label, x, y, w, h in boxes:
        fill, edge = node_color(label)
        shadow = FancyBboxPatch(
            (x + 0.035, y - 0.035),
            w,
            h,
            boxstyle="round,pad=0.035,rounding_size=0.05",
            linewidth=0,
            facecolor="#d7dde6",
            alpha=0.65,
            zorder=1,
        )
        rect = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.035,rounding_size=0.05",
            linewidth=1.6,
            edgecolor=edge,
            facecolor=fill,
            zorder=2,
        )
        ax.add_patch(shadow)
        ax.add_patch(rect)
        ax.text(
            x + w / 2,
            y + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=9.8,
            color="#17212b",
            wrap=True,
            zorder=3,
        )
    for start, end in arrows:
        _, x1, y1, w1, h1 = boxes[start]
        _, x2, y2, _, h2 = boxes[end]
        arrow = FancyArrowPatch(
            (x1 + w1, y1 + h1 / 2),
            (x2, y2 + h2 / 2),
            arrowstyle="->",
            mutation_scale=15,
            linewidth=1.8,
            color="#34495e",
            shrinkA=8,
            shrinkB=8,
            zorder=0,
        )
        ax.add_patch(arrow)
    save_plot(path)


def draw_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    fill: str,
    edge: str,
    fontsize: float = 9.2,
    linewidth: float = 1.6,
    linestyle: str = "-",
) -> None:
    shadow = FancyBboxPatch(
        (x + 0.035, y - 0.035),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.05",
        linewidth=0,
        facecolor="#d7dde6",
        alpha=0.6,
        zorder=1,
    )
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.05",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=fill,
        linestyle=linestyle,
        zorder=2,
    )
    ax.add_patch(shadow)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, color="#17212b", zorder=3)


def draw_arrow(ax, start: tuple[float, float], end: tuple[float, float], dashed: bool = False, color: str = "#34495e") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="->",
        mutation_scale=14,
        linewidth=1.6,
        color=color,
        linestyle="--" if dashed else "-",
        shrinkA=6,
        shrinkB=6,
        zorder=0,
    )
    ax.add_patch(arrow)


def draw_feature_stack(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    fill: str,
    edge: str,
    depth: int = 3,
    fontsize: float = 8.6,
) -> None:
    for idx in range(depth - 1, -1, -1):
        dx = idx * 0.055
        dy = idx * 0.045
        patch = FancyBboxPatch(
            (x + dx, y + dy),
            w,
            h,
            boxstyle="round,pad=0.025,rounding_size=0.035",
            linewidth=1.0 if idx else 1.6,
            edgecolor=edge,
            facecolor=fill,
            alpha=0.62 + 0.12 * (depth - idx),
            zorder=2 + idx,
        )
        ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fontsize, color="#17212b", zorder=8)


def draw_legend(ax, x: float, y: float, items: list[tuple[str, str, str]]) -> None:
    for idx, (label, fill, edge) in enumerate(items):
        yy = y - idx * 0.28
        patch = FancyBboxPatch(
            (x, yy),
            0.18,
            0.14,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.0,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(patch)
        ax.text(x + 0.25, yy + 0.07, label, va="center", fontsize=7.5, color="#263238")


def draw_dcgan_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 4.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("DCGAN generator", fontsize=17, pad=12, fontweight="bold", color="#263238")

    draw_box(ax, 0.25, 1.55, 0.85, 0.7, "Latent\nz=128", "#f5f7fa", "#2d4059")
    draw_box(ax, 1.55, 1.42, 1.1, 0.95, "Linear\nBatchNorm\nReLU", "#eef1fb", "#4a5f9e", fontsize=8.2)
    stacks = [
        (3.05, 1.44, 1.08, 0.88, "4x4\n512", "#eef1fb", "#4a5f9e"),
        (4.65, 1.36, 1.15, 1.02, "8x8\n256", "#e8f3f1", "#2f6f63"),
        (6.32, 1.28, 1.22, 1.18, "16x16\n128", "#e8f3f1", "#2f6f63"),
        (8.06, 1.18, 1.30, 1.36, "32x32\n64", "#e8f3f1", "#2f6f63"),
        (9.9, 1.08, 1.42, 1.55, "64x64\nRGB", "#f4ecf7", "#7b4b8d"),
    ]
    for x, y, w, h, label, fill, edge in stacks:
        draw_feature_stack(ax, x, y, w, h, label, fill, edge)

    arrow_edges = [
        ((1.1, 1.9), (1.55, 1.9)),
        ((2.65, 1.9), (3.05, 1.9)),
        ((4.13, 1.9), (4.65, 1.9)),
        ((5.8, 1.9), (6.32, 1.9)),
        ((7.54, 1.9), (8.06, 1.9)),
        ((9.36, 1.9), (9.9, 1.9)),
    ]
    for start, end in arrow_edges:
        draw_arrow(ax, start, end)

    ax.text(5.95, 0.72, "ConvTranspose2d + BatchNorm + ReLU", ha="center", fontsize=8.5, color="#455a64")
    ax.text(10.6, 0.72, "Tanh output", ha="center", fontsize=8.5, color="#455a64")
    draw_legend(
        ax,
        0.35,
        0.45,
        [
            ("latent/projection", "#eef1fb", "#4a5f9e"),
            ("feature maps", "#e8f3f1", "#2f6f63"),
            ("RGB output", "#f4ecf7", "#7b4b8d"),
        ],
    )
    save_plot(ASSETS / "dcgan_architecture.png")


def draw_stylegan_architecture(path: Path, title: str, two_convs: bool) -> None:
    fig, ax = plt.subplots(figsize=(13.8, 5.2))
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title(title, fontsize=17, pad=12, fontweight="bold", color="#263238")

    draw_box(ax, 0.25, 3.65, 0.7, 0.52, "z", "#f5f7fa", "#2d4059")
    draw_box(ax, 1.25, 3.5, 1.35, 0.82, "Mapping\n4-layer MLP", "#eef1fb", "#4a5f9e", fontsize=8.8)
    draw_box(ax, 2.95, 3.65, 0.7, 0.52, "w", "#f5f7fa", "#2d4059")
    draw_arrow(ax, (0.95, 3.91), (1.25, 3.91))
    draw_arrow(ax, (2.6, 3.91), (2.95, 3.91))

    # Style bus: w controls every synthesis block through AdaIN.
    ax.plot([3.65, 12.1], [4.42, 4.42], color="#7b4b8d", linewidth=1.8, linestyle="--")
    ax.text(7.35, 4.58, "style control via AdaIN", ha="center", fontsize=8.8, color="#7b4b8d")
    draw_arrow(ax, (3.35, 3.92), (3.8, 4.42), dashed=True, color="#7b4b8d")

    draw_box(ax, 0.9, 1.18, 1.35, 0.82, "Learned\n4x4 constant", "#f5f7fa", "#2d4059", fontsize=8.8)

    resolutions = ["4x4", "8x8", "16x16", "32x32", "64x64"]
    channels = ["512", "256", "128", "64", "32"]
    x0 = 3.0
    block_w = 1.35
    gap = 0.58
    prev_right = 2.25
    for idx, (resolution, channel) in enumerate(zip(resolutions, channels)):
        x = x0 + idx * (block_w + gap)
        text = ("2 x StyledConv" if two_convs else "StyledConv") + f"\n{resolution}\nC={channel}"
        fill = "#eef1fb" if idx == 0 else "#e8f3f1"
        edge = "#4a5f9e" if idx == 0 else "#2f6f63"
        draw_feature_stack(ax, x, 1.25, block_w, 1.08, text, fill, edge, depth=2 if two_convs else 1, fontsize=7.8)
        draw_arrow(ax, (prev_right, 1.78), (x, 1.78))
        prev_right = x + block_w
        # Per-layer noise and AdaIN hints.
        ax.plot([x + block_w / 2, x + block_w / 2], [4.42, 2.38], color="#7b4b8d", linewidth=1.1, linestyle="--")
        draw_arrow(ax, (x + block_w / 2, 4.42), (x + block_w / 2, 2.38), dashed=True, color="#7b4b8d")
        ax.text(x + block_w / 2, 0.9, "noise", ha="center", fontsize=7.3, color="#a46c12")
        draw_arrow(ax, (x + block_w / 2, 1.02), (x + block_w / 2, 1.22), dashed=True, color="#a46c12")
        if idx > 0:
            ax.text(x - 0.29, 2.55, "upsample", rotation=25, fontsize=7.2, color="#455a64")

    draw_box(ax, 12.55, 1.38, 1.1, 0.8, "ToRGB\nTanh", "#f4ecf7", "#7b4b8d")
    draw_arrow(ax, (prev_right, 1.78), (12.55, 1.78))
    draw_box(ax, 12.55, 0.45, 1.1, 0.55, "RGB\n64x64", "#f4ecf7", "#7b4b8d", fontsize=8.4)
    draw_arrow(ax, (13.1, 1.38), (13.1, 1.0))

    draw_legend(
        ax,
        0.35,
        0.55,
        [
            ("mapping/style", "#eef1fb", "#4a5f9e"),
            ("synthesis feature block", "#e8f3f1", "#2f6f63"),
            ("style/noise branch", "#fff4df", "#a46c12"),
            ("RGB output", "#f4ecf7", "#7b4b8d"),
        ],
    )
    save_plot(path)


def make_diagrams() -> None:
    draw_flow(
        ASSETS / "project_pipeline.png",
        [
            ("CelebA\naligned faces", 0.15, 1.5, 1.45, 0.8),
            ("Preprocess\n64x64, [-1,1]", 2.05, 1.5, 1.55, 0.8),
            ("Train models\nDCGAN\nStyleGAN-lite\nStyleGAN-lite-v2", 4.15, 1.35, 2.55, 1.1),
            ("Samples\nand interpolation", 7.2, 2.3, 1.6, 0.8),
            ("FID / IS\nMS-SSIM", 7.2, 0.7, 1.6, 0.8),
            ("Report\nanalysis", 9.1, 1.5, 1.25, 0.8),
        ],
        [(0, 1), (1, 2), (2, 3), (2, 4), (3, 5), (4, 5)],
        "Project pipeline",
        x_max=10.6,
    )
    draw_flow(
        ASSETS / "gan_training_diagram.png",
        [
            ("Latent z", 0.3, 2.3, 1.2, 0.7),
            ("Generator G", 2.0, 2.3, 1.5, 0.7),
            ("Fake image", 4.0, 2.3, 1.4, 0.7),
            ("Real image", 4.0, 0.7, 1.4, 0.7),
            ("Discriminator D", 6.0, 1.5, 1.7, 0.8),
            ("Adversarial\nlosses", 8.2, 1.5, 1.4, 0.8),
        ],
        [(0, 1), (1, 2), (2, 4), (3, 4), (4, 5)],
        "GAN adversarial training",
    )
    draw_dcgan_architecture()
    draw_stylegan_architecture(ASSETS / "stylegan_lite_architecture.png", "StyleGAN-lite generator", two_convs=False)
    draw_stylegan_architecture(
        ASSETS / "stylegan_lite_v2_architecture.png",
        "StyleGAN-lite-v2 generator",
        two_convs=True,
    )
    draw_flow(
        ASSETS / "evaluation_pipeline.png",
        [
            ("Real images", 0.2, 2.3, 1.2, 0.7),
            ("Inception\nfeatures", 1.8, 2.3, 1.4, 0.7),
            ("Cached real\nstatistics", 3.6, 2.3, 1.4, 0.7),
            ("Generator", 0.2, 0.8, 1.2, 0.7),
            ("Fake images", 1.8, 0.8, 1.4, 0.7),
            ("FID / IS /\nMS-SSIM", 5.6, 1.5, 1.6, 0.8),
            ("JSON result\nand report", 7.8, 1.5, 1.5, 0.8),
        ],
        [(0, 1), (1, 2), (3, 4), (2, 5), (4, 5), (5, 6)],
        "Evaluation pipeline",
    )
    draw_flow(
        ASSETS / "improvement_path.png",
        [
            ("Observed issue\nmode collapse", 0.2, 1.5, 1.5, 0.8),
            ("Stronger synthesis\n2 StyledConv", 2.0, 2.3, 1.5, 0.8),
            ("TTUR + R1\nstable D", 4.0, 2.3, 1.5, 0.8),
            ("Minibatch stddev\npenalize repetition", 2.0, 0.7, 1.7, 0.8),
            ("EMA\nstable samples", 4.3, 0.7, 1.3, 0.8),
            ("StyleGAN-lite-v2\nFID 12.78", 6.5, 1.5, 1.6, 0.8),
        ],
        [(0, 1), (1, 2), (0, 3), (3, 4), (2, 5), (4, 5)],
        "Stabilization path",
    )


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    logs = {name: read_jsonl(path) for name, path in LOGS.items()}
    evals = {name: read_eval(path) for name, path in EVALS.items()}
    plot_fid_curve(logs)
    plot_losses(logs)
    plot_diversity(logs["StyleGAN-lite-v2"])
    plot_final_bars(evals)
    plot_metrics_dashboard(logs, evals)
    make_image_composites()
    make_diagrams()
    print(f"Saved report figures to {ASSETS}")


if __name__ == "__main__":
    main()
