import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

def simulate_tremor(image_arr, frequency=0.08, amplitude=2.5):
    """
    Simulates neuromuscular tremor using sinusoidal coordinate displacement 
    and micro-jitter without requiring SciPy DLLs.
    """
    h, w = image_arr.shape
    y_coords, x_coords = np.indices((h, w), dtype=np.float32)

    # Multi-frequency sinusoidal tremor along x and y axes
    dx = amplitude * np.sin(2 * np.pi * frequency * y_coords) + np.random.normal(0, 0.4, (h, w))
    dy = amplitude * np.cos(2 * np.pi * frequency * x_coords) + np.random.normal(0, 0.4, (h, w))

    # Apply coordinate displacement with boundary clamping
    map_x = np.clip(np.round(x_coords + dx).astype(int), 0, w - 1)
    map_y = np.clip(np.round(y_coords + dy).astype(int), 0, h - 1)

    return image_arr[map_y, map_x]

def simulate_pressure_variation(image_arr, threshold=128):
    """
    Simulates variable grip pressure: ink pooling and faint/broken strokes 
    using PIL min/max morphological filters.
    """
    pil_img = Image.fromarray(image_arr.astype(np.uint8))
    
    # MinFilter dilates dark stroke regions (heavy pressure / ink pooling)
    dilated = np.array(pil_img.filter(ImageFilter.MinFilter(size=3)), dtype=np.float32)
    # MaxFilter erodes dark strokes (light pressure / broken stroke)
    eroded = np.array(pil_img.filter(ImageFilter.MaxFilter(size=3)), dtype=np.float32)

    stroke_mask = image_arr < threshold
    noise = np.random.rand(*image_arr.shape)

    result = image_arr.copy()
    # Heavy pressure spots
    result[stroke_mask & (noise > 0.65)] = dilated[stroke_mask & (noise > 0.65)]
    # Faded / fragmented spots
    result[stroke_mask & (noise < 0.25)] = eroded[stroke_mask & (noise < 0.25)]

    return result

def apply_dysgraphia_simulation(img_path, severity=1.0):
    pil_img = Image.open(img_path).convert('L')
    img_arr = np.array(pil_img, dtype=np.float32)

    # 1. Apply motor tremor
    tremor_img = simulate_tremor(img_arr, frequency=0.07, amplitude=2.5 * severity)

    # 2. Apply pressure fluctuations & stroke breaks
    perturbed_img = simulate_pressure_variation(tremor_img)

    return img_arr, perturbed_img

if __name__ == "__main__":
    search_paths = [
        os.path.join("..", "data", "**", "*.png"),
        os.path.join("data", "**", "*.png"),
        os.path.join("..", "..", "iamonDB", "**", "*.png")
    ]
    
    png_files = []
    for pattern in search_paths:
        png_files.extend(glob.glob(pattern, recursive=True))

    if not png_files:
        print("No .png files found. Verify that the PNG files extracted into 'data' or 'iamonDB'.")
    else:
        sample_path = png_files[0]
        print(f"Applying perturbation to: {sample_path}")
        
        original, simulated = apply_dysgraphia_simulation(sample_path, severity=1.2)

        fig, axes = plt.subplots(2, 1, figsize=(8, 4))
        axes[0].imshow(original, cmap='gray')
        axes[0].set_title("Original Handwriting Sample")
        axes[0].axis('off')

        axes[1].imshow(simulated, cmap='gray')
        axes[1].set_title("Simulated Dysgraphic Perturbation (Tremor + Irregular Pressure)")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()