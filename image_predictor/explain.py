#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAM/Grad-CAM export utility for ectopia lentis severity figures.
"""

import argparse
import hashlib
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

try:
    import shap
except ImportError:
    shap = None

from predict import (
    SUPPORTED_EXTENSIONS,
    get_default_model_path,
    get_transform,
    load_model,
)


DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "output",
    "explanations",
)

CLASS_LABELS_EN = {
    0: "Mild",
    1: "Moderate",
    2: "Severe",
}


def normalize_map(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    cam -= cam.min()
    max_value = cam.max()
    if max_value > 0:
        cam /= max_value
    return cam


class CAMGenerator:
    """CAM-style explainer for the current classifier head using local linearization in eval mode."""

    def __init__(self, model: torch.nn.Module):
        self.model = model

    @staticmethod
    def _bn_scale_shift(bn: torch.nn.BatchNorm1d) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        shift = bn.bias - bn.running_mean * scale
        return scale.detach(), shift.detach()

    def _forward_features(self, image_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.model.features(image_tensor)
        pooled = self.model.adaptive_pool(feature_map)
        return feature_map, pooled

    def generate(self, image_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, torch.Tensor]:
        with torch.no_grad():
            feature_map, pooled = self._forward_features(image_tensor)
            x = pooled.flatten(1)

            linear1 = self.model.classifier[2]
            bn1 = self.model.classifier[3]
            linear2 = self.model.classifier[6]
            bn2 = self.model.classifier[7]
            linear3 = self.model.classifier[10]

            z1 = linear1(x)
            scale1, shift1 = self._bn_scale_shift(bn1)
            z1_bn = z1 * scale1.unsqueeze(0) + shift1.unsqueeze(0)
            gate1 = (z1_bn > 0).to(z1_bn.dtype)[0]

            a1 = F.relu(z1_bn)
            z2 = linear2(a1)
            scale2, shift2 = self._bn_scale_shift(bn2)
            z2_bn = z2 * scale2.unsqueeze(0) + shift2.unsqueeze(0)
            gate2 = (z2_bn > 0).to(z2_bn.dtype)[0]

            a2 = F.relu(z2_bn)
            logits = linear3(a2)
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1).item())

            w3 = linear3.weight[target_class].detach()
            v2 = w3 * gate2 * scale2
            v1 = torch.matmul(v2, linear2.weight.detach())
            v1 = v1 * gate1 * scale1
            w_eff = torch.matmul(v1, linear1.weight.detach())

            pooled_single = pooled[0]
            spatial_weights = w_eff.view_as(pooled_single)
            cam = (spatial_weights * pooled_single).sum(dim=0)
            cam = F.relu(cam)
            cam = normalize_map(cam.cpu().numpy())
            return cam, logits.detach()


class GradCAMGenerator:
    """Minimal Grad-CAM implementation for the last convolutional feature map."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = [
            self.target_layer.register_forward_hook(self._forward_hook),
        ]

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()
        output.register_hook(self._gradient_hook)

    def _gradient_hook(self, grad):
        self.gradients = grad.detach()

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def generate(self, image_tensor: torch.Tensor, target_class: Optional[int] = None) -> Tuple[np.ndarray, torch.Tensor]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        if target_class is None:
            target_class = int(torch.argmax(logits, dim=1).item())

        score = logits[:, target_class].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        pooled_gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (pooled_gradients * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().detach().cpu().numpy()
        cam = normalize_map(cam)
        return cam, logits.detach()


class SHAPGradientGenerator:
    """Image-level SHAP explainer using GradientExplainer and a small background set."""

    def __init__(self, model: torch.nn.Module, background: torch.Tensor, nsamples: int):
        if shap is None:
            raise ImportError("shap is required for --method shap. Install it with `pip install shap`.")
        self.model = model
        self.background = background
        self.nsamples = nsamples
        self.explainer = shap.GradientExplainer(model, background)

    @staticmethod
    def _extract_selected_values(
        shap_values: Any,
        target_class: int,
        image_tensor: torch.Tensor,
    ) -> np.ndarray:
        if isinstance(shap_values, list):
            selected = shap_values[target_class]
        else:
            values = np.asarray(shap_values)
            if values.ndim < 4:
                raise ValueError(f"Unsupported SHAP output shape: {values.shape}")
            if values.shape[-1] == image_tensor.shape[1]:
                selected = values[..., target_class]
            elif values.ndim >= 5 and values.shape[1] == image_tensor.shape[1]:
                selected = values[target_class]
            elif values.shape[-1] > target_class:
                selected = values[..., target_class]
            else:
                raise ValueError(f"Unsupported SHAP output shape: {values.shape}")
        selected = np.asarray(selected)
        if selected.ndim == 4:
            selected = selected[0]
        return selected

    @staticmethod
    def _to_hwc(values: np.ndarray) -> np.ndarray:
        if values.ndim != 3:
            raise ValueError(f"Expected 3D SHAP values, got shape {values.shape}")
        if values.shape[0] in {1, 3} and values.shape[0] != values.shape[-1]:
            return np.transpose(values, (1, 2, 0))
        return values

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, torch.Tensor, Dict[str, Any]]:
        with torch.no_grad():
            logits = self.model(image_tensor)
            if target_class is None:
                target_class = int(torch.argmax(logits, dim=1).item())

        shap_values = self.explainer.shap_values(image_tensor, nsamples=self.nsamples)
        selected = self._extract_selected_values(shap_values, target_class, image_tensor)
        selected_hwc = self._to_hwc(selected)
        attribution = np.abs(selected_hwc).mean(axis=2)
        attribution = normalize_map(attribution)
        return attribution, logits.detach(), {"shap_values_hwc": selected_hwc}


def collect_image_files(folder_path: str) -> List[str]:
    files = []
    for filename in sorted(os.listdir(folder_path)):
        ext = os.path.splitext(filename)[1]
        if ext in SUPPORTED_EXTENSIONS:
            files.append(os.path.join(folder_path, filename))
    return files


def load_original_image(image_path: str) -> Image.Image:
    return Image.open(image_path).convert("RGB")


def prepare_input(image: Image.Image, transform, device: torch.device) -> torch.Tensor:
    return transform(image).unsqueeze(0).to(device)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype, device=tensor.device).view(1, 3, 1, 1)
    image = tensor.detach().clone() * std + mean
    image = image.clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy()
    return np.uint8(image * 255.0)


def to_grayscale_rgb(image: np.ndarray) -> np.ndarray:
    gray = np.dot(image[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    gray = np.clip(gray, 0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def prepare_cam_mask(
    cam: np.ndarray,
    image_size: Tuple[int, int],
    threshold: float,
    blur_radius: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    resized = Image.fromarray(np.uint8(cam * 255)).resize(image_size, Image.BILINEAR)
    if blur_radius is None:
        blur_radius = max(min(image_size) / 224.0, 1.2)
    resized = resized.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    cam_resized = np.asarray(resized).astype(np.float32) / 255.0

    high = np.percentile(cam_resized, 99.0)
    if high > 0:
        cam_resized = np.clip(cam_resized / high, 0.0, 1.0)

    mask = np.where(cam_resized >= threshold, cam_resized, 0.0)
    return cam_resized, mask


def build_heatmap(gray_base: np.ndarray, cam: np.ndarray, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
    cam_resized, mask = prepare_cam_mask(cam, (gray_base.shape[1], gray_base.shape[0]), threshold=threshold)
    cam_uint8 = np.uint8(np.clip(mask, 0.0, 1.0) * 255.0)
    colored = plt.cm.jet(cam_uint8.astype(np.float32) / 255.0)[..., :3] * 255.0
    heatmap = gray_base.astype(np.float32) * 0.5 + colored.astype(np.float32) * 0.5
    zero_mask = cam_uint8 == 0
    heatmap[zero_mask] = gray_base.astype(np.float32)[zero_mask]
    return np.clip(heatmap, 0, 255).astype(np.uint8), mask


def build_overlay(gray_base: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    cam_uint8 = np.uint8(np.clip(mask, 0.0, 1.0) * 255.0)
    colored = plt.cm.jet(cam_uint8.astype(np.float32) / 255.0)[..., :3] * 255.0
    overlay = gray_base.astype(np.float32) * (1.0 - alpha) + colored.astype(np.float32) * alpha
    zero_mask = cam_uint8 == 0
    overlay[zero_mask] = gray_base.astype(np.float32)[zero_mask]
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_rgb_image(array: np.ndarray, output_path: str):
    Image.fromarray(array).save(output_path)


def save_shap_plot(
    shap_values_hwc: np.ndarray,
    image_rgb: np.ndarray,
    class_name: str,
    output_path: str,
    dpi: int,
):
    if shap is None:
        raise ImportError("shap is required to save SHAP plots.")
    pixel_values = image_rgb.astype(np.float32) / 255.0
    shap.image_plot(
        [shap_values_hwc[np.newaxis, ...]],
        pixel_values[np.newaxis, ...],
        labels=np.array([[class_name]]),
        show=False,
    )
    plt.gcf().savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close("all")


def format_probability_lines(probabilities: np.ndarray) -> List[str]:
    return [f"{CLASS_LABELS_EN[i]}: {probabilities[i]:.4f}" for i in range(len(CLASS_LABELS_EN))]


def save_panel(
    original: np.ndarray,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    pred_label: str,
    probabilities: np.ndarray,
    output_path: str,
    dpi: int,
    method_name: str,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        (original, "Original"),
        (heatmap, f"{method_name} Heatmap"),
        (overlay, "Overlay"),
        (original, "Prediction"),
    ]

    for ax, (image, title) in zip(axes.flat, panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    axes[1, 1].text(
        0.03,
        0.92,
        f"Prediction: {pred_label}\n\n" + "\n".join(format_probability_lines(probabilities)),
        transform=axes[1, 1].transAxes,
        va="top",
        ha="left",
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "black", "boxstyle": "round,pad=0.5"},
    )
    axes[1, 1].imshow(original)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def explain_image(
    model: torch.nn.Module,
    cam_generator,
    image_path: str,
    transform,
    device: torch.device,
    output_dir: str,
    alpha: float,
    dpi: int,
    threshold: float,
    target_class: Optional[int],
    method_name: str,
) -> Dict[str, Any]:
    original_image = load_original_image(image_path)
    input_tensor = prepare_input(original_image, transform, device)
    generated = cam_generator.generate(input_tensor, target_class=target_class)
    extra_outputs: Dict[str, Any] = {}
    if len(generated) == 3:
        cam, logits, extra_outputs = generated
    else:
        cam, logits = generated
    input_rgb = denormalize_image(input_tensor)

    probabilities = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_class = int(np.argmax(probabilities))
    explained_class = pred_class if target_class is None else int(target_class)
    pred_label = CLASS_LABELS_EN[pred_class]
    image_name = os.path.basename(image_path)
    case_id = sanitize_stem(image_path)

    os.makedirs(output_dir, exist_ok=True)

    original_np = input_rgb
    gray_base = to_grayscale_rgb(input_rgb)
    heatmap_np, mask = build_heatmap(gray_base, cam, threshold=threshold)
    overlay_np = build_overlay(gray_base, mask, alpha=alpha)

    original_path = os.path.join(output_dir, "original.png")
    heatmap_path = os.path.join(output_dir, "heatmap.png")
    overlay_path = os.path.join(output_dir, "overlay.png")
    panel_path = os.path.join(output_dir, "figure_panel.png")
    shap_plot_path = os.path.join(output_dir, "shap_plot.png")

    save_rgb_image(original_np, original_path)
    save_rgb_image(heatmap_np, heatmap_path)
    save_rgb_image(overlay_np, overlay_path)
    if "shap_values_hwc" in extra_outputs:
        save_shap_plot(
            shap_values_hwc=extra_outputs["shap_values_hwc"],
            image_rgb=original_np,
            class_name=CLASS_LABELS_EN[explained_class],
            output_path=shap_plot_path,
            dpi=dpi,
        )
    save_panel(
        original=original_np,
        heatmap=heatmap_np,
        overlay=overlay_np,
        pred_label=f"{pred_label} (Explained Class: {CLASS_LABELS_EN[explained_class]})",
        probabilities=probabilities,
        output_path=panel_path,
        dpi=dpi,
        method_name=method_name,
    )

    result = {
        "case_id": case_id,
        "predicted_label": pred_label,
        "explained_class": CLASS_LABELS_EN[explained_class],
        "mild_probability": float(probabilities[0]),
        "moderate_probability": float(probabilities[1]),
        "severe_probability": float(probabilities[2]),
        "original_path": original_path,
        "heatmap_path": heatmap_path,
        "overlay_path": overlay_path,
        "panel_path": panel_path,
    }
    if "shap_values_hwc" in extra_outputs:
        result["shap_plot_path"] = shap_plot_path
    return result


def get_default_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sanitize_stem(path: str) -> str:
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", stem)
    stem = stem.strip("_")
    digest = hashlib.md5(os.path.basename(path).encode("utf-8")).hexdigest()[:6]
    prefix = stem or "case"
    return f"{prefix}_{digest}"


def build_background_tensor(
    image_paths: List[str],
    transform,
    device: torch.device,
    max_images: int,
) -> torch.Tensor:
    selected_paths = image_paths[:max_images]
    tensors = []
    for path in selected_paths:
        image = load_original_image(path)
        tensors.append(transform(image))
    if not tensors:
        raise ValueError("No background images available for SHAP.")
    return torch.stack(tensors, dim=0).to(device)


def main():
    parser = argparse.ArgumentParser(
        description="CAM/Grad-CAM export utility for ectopia lentis severity figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single image:
    python explain.py --image test.jpg

  Folder:
    python explain.py --folder ./images

  Custom output directory:
    python explain.py --folder ./images --output-dir ./output/explanations

  Specific explained class:
    python explain.py --image test.jpg --target-class 2

  Use Grad-CAM instead of CAM:
    python explain.py --image test.jpg --method gradcam

  Use SHAP instead of CAM:
    python explain.py --image test.jpg --method shap
        """,
    )

    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--folder", type=str, help="Path to an image folder")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path (default: weights/best_model.pth)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Execution device: cuda or cpu (default: auto)")
    parser.add_argument("--method", type=str, choices=["cam", "gradcam", "shap"], default="cam", help="Explanation method, default cam")
    parser.add_argument("--target-class", type=int, choices=[0, 1, 2], default=None, help="Explained class, default is predicted class")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha, default 0.5")
    parser.add_argument("--dpi", type=int, default=1000, help="Export DPI for the panel figure, default 1000")
    parser.add_argument("--cam-threshold", type=float, default=0.6, help="Threshold for suppressing low-response regions")
    parser.add_argument("--shap-background-dir", type=str, default=None, help="Folder used to sample SHAP background images")
    parser.add_argument("--shap-background-size", type=int, default=8, help="Number of background images for SHAP")
    parser.add_argument("--shap-nsamples", type=int, default=200, help="Number of samples used by SHAP GradientExplainer")

    args = parser.parse_args()

    if args.image is None and args.folder is None:
        parser.print_help()
        print("\nError: please provide --image or --folder")
        sys.exit(1)

    if args.alpha < 0 or args.alpha > 1:
        raise ValueError("--alpha must be between 0 and 1")
    if args.cam_threshold < 0 or args.cam_threshold > 1:
        raise ValueError("--cam-threshold must be between 0 and 1")

    device = get_default_device(args.device)
    print(f"Device: {device}")

    model_path = args.model if args.model else get_default_model_path()
    model = load_model(model_path, device)
    transform = get_transform()
    if args.method == "shap":
        background_source = args.shap_background_dir
        if background_source is None:
            background_source = args.folder if args.folder else os.path.dirname(os.path.abspath(args.image))
        if not os.path.isdir(background_source):
            raise NotADirectoryError(f"SHAP background folder not found: {background_source}")
        background_images = collect_image_files(background_source)
        if not background_images:
            raise FileNotFoundError(f"No supported background images found in: {background_source}")
        background_tensor = build_background_tensor(
            image_paths=background_images,
            transform=transform,
            device=device,
            max_images=max(1, args.shap_background_size),
        )
        cam_generator = SHAPGradientGenerator(model, background_tensor, nsamples=args.shap_nsamples)
    else:
        cam_generator = CAMGenerator(model) if args.method == "cam" else GradCAMGenerator(model, model.features[-1])
    method_name = "Grad-CAM" if args.method == "gradcam" else args.method.upper()

    try:
        if args.image:
            if not os.path.exists(args.image):
                raise FileNotFoundError(f"Image file not found: {args.image}")

            output_dir = os.path.join(args.output_dir, sanitize_stem(args.image))
            result = explain_image(
                model=model,
                cam_generator=cam_generator,
                image_path=args.image,
                transform=transform,
                device=device,
                output_dir=output_dir,
                alpha=args.alpha,
                dpi=args.dpi,
                threshold=args.cam_threshold,
                target_class=args.target_class,
                method_name=method_name,
            )
            print("\nExport complete:")
            print(f"  Prediction: {result['predicted_label']}")
            print(f"  Original: {result['original_path']}")
            print(f"  Heatmap: {result['heatmap_path']}")
            print(f"  Overlay: {result['overlay_path']}")
            print(f"  Figure Panel: {result['panel_path']}")
            if "shap_plot_path" in result:
                print(f"  SHAP Plot: {result['shap_plot_path']}")

        else:
            if not os.path.isdir(args.folder):
                raise NotADirectoryError(f"Folder not found: {args.folder}")

            image_files = collect_image_files(args.folder)
            if not image_files:
                raise FileNotFoundError(f"No supported image files found in: {args.folder}")

            os.makedirs(args.output_dir, exist_ok=True)
            rows = []
            print(f"Found {len(image_files)} images. Generating {method_name} outputs...")
            for index, image_path in enumerate(image_files, start=1):
                image_output_dir = os.path.join(args.output_dir, sanitize_stem(image_path))
                row = explain_image(
                    model=model,
                    cam_generator=cam_generator,
                    image_path=image_path,
                    transform=transform,
                    device=device,
                    output_dir=image_output_dir,
                    alpha=args.alpha,
                    dpi=args.dpi,
                    threshold=args.cam_threshold,
                    target_class=args.target_class,
                    method_name=method_name,
                )
                rows.append(row)
                print(f"  [{index}/{len(image_files)}] {row['case_id']} -> {row['predicted_label']}")

            summary_path = os.path.join(args.output_dir, "summary.csv")
            pd.DataFrame(rows).to_csv(summary_path, index=False, encoding="utf-8-sig")
            print(f"\nSummary saved to: {summary_path}")

    finally:
        if hasattr(cam_generator, "remove"):
            cam_generator.remove()


if __name__ == "__main__":
    main()
