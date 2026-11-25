import os
import glob
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def simulate_tremor(image_arr, frequency=0.08, amplitude=2.5):
    h, w = image_arr.shape
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)

    dx = amplitude * np.sin(2 * np.pi * frequency * y_coords) + np.random.normal(0, 0.35, (h, w))
    dy = amplitude * np.cos(2 * np.pi * frequency * x_coords) + np.random.normal(0, 0.35, (h, w))

    map_x = np.clip(np.round(x_coords + dx).astype(int), 0, w - 1)
    map_y = np.clip(np.round(y_coords + dy).astype(int), 0, h - 1)

    return image_arr[map_y, map_x]


def simulate_pressure_variation(image_arr, threshold=128):
    pil_img = Image.fromarray(image_arr.astype(np.uint8))

    dilated = np.array(pil_img.filter(ImageFilter.MinFilter(size=3)), dtype=np.float32)
    eroded = np.array(pil_img.filter(ImageFilter.MaxFilter(size=3)), dtype=np.float32)

    stroke_mask = image_arr < threshold
    noise = np.random.rand(*image_arr.shape)

    result = image_arr.copy()
    result[stroke_mask & (noise > 0.65)] = dilated[stroke_mask & (noise > 0.65)]
    result[stroke_mask & (noise < 0.25)] = eroded[stroke_mask & (noise < 0.25)]
    return result


def perturb_single_image(img_path, severity):
    pil_img = Image.open(img_path).convert('L')
    img_arr = np.array(pil_img, dtype=np.float32)

    # Vary frequency and amplitude dynamically with severity
    freq = random.uniform(0.06, 0.10)
    amp = random.uniform(1.8, 3.2) * severity

    tremor_img = simulate_tremor(img_arr, frequency=freq, amplitude=amp)
    final_img = simulate_pressure_variation(tremor_img)
    return Image.fromarray(final_img.astype(np.uint8))


def apply_scanner_variation(pil_img):
    # Slight random brightness & contrast to mimic real paper/scanner lighting
    pil_img = ImageEnhance.Brightness(pil_img).enhance(random.uniform(0.92, 1.08))
    pil_img = ImageEnhance.Contrast(pil_img).enhance(random.uniform(0.92, 1.08))
    return pil_img


def run_batch_generation(num_samples=2500):
    """
    Finds raw images, samples a target batch size, and generates
    paired control vs. dysgraphic image sets with realistic scanner variation.
    """
    input_paths = [
        os.path.join("..", "data", "**", "*.png"),
        os.path.join("..", "..", "iamonDB", "**", "*.png")
    ]

    all_files = []
    for pattern in input_paths:
        all_files.extend(glob.glob(pattern, recursive=True))

    # Filter out any files already inside a processed folder
    valid_files = [f for f in all_files if "processed" not in f]

    if not valid_files:
        print("No source PNG files found to process.")
        return

    # Set seeds for reproducibility and select target subset
    random.seed(42)
    np.random.seed(42)
    random.shuffle(valid_files)
    selected_files = valid_files[:num_samples]

    out_control = Path("../data/processed/control")
    out_dysgraphic = Path("../data/processed/dysgraphic")
    out_control.mkdir(parents=True, exist_ok=True)
    out_dysgraphic.mkdir(parents=True, exist_ok=True)

    print(f"Starting batch generation for {len(selected_files)} samples...")

    for idx, file_path in enumerate(selected_files, 1):
        filename = f"sample_{idx:05d}.png"

        # 1. Save original as control (with scanner variation)
        raw_img = Image.open(file_path).convert('L')
        apply_scanner_variation(raw_img).save(out_control / filename)

        # 2. Apply random severity (0.8 to 1.5) and save as dysgraphic (with scanner variation)
        severity = random.uniform(0.8, 1.5)
        dys_img = perturb_single_image(file_path, severity=severity)
        apply_scanner_variation(dys_img).save(out_dysgraphic / filename)

        if idx % 250 == 0 or idx == len(selected_files):
            print(f"Processed [{idx}/{len(selected_files)}] pairs.")

    print("\nBatch generation complete!")
    print(f"Control directory: {out_control.resolve()}")
    print(f"Dysgraphic directory: {out_dysgraphic.resolve()}")


if __name__ == "__main__":
    run_batch_generation(num_samples=2500)