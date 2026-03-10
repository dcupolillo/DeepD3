"""Command-line interface for DeepD3 inference.

This script enables running DeepD3 inference as a subprocess
from external environments without requiring DeepD3 installation.
"""
from __future__ import annotations
import argparse
import numpy as np
from pathlib import Path
import tifffile
from deepd3.inference.detection import inference



def main():
    """
    Main function to parse command-line arguments and run DeepD3 inference.
    
    """
    
    parser = argparse.ArgumentParser(
        description='Run DeepD3 spine/dendrite segmentation'
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        type=str or Path,
        help='Path to input directory containing .tif images'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        type=str or Path,
        help='Path to output directory for predictions'
    )

    parser.add_argument(
        '-m', '--model',
        required=True,
        type=str or Path,
        help='Path to DeepD3 model file (.h5)'
    )

    parser.add_argument(
        '--device',
        default='/GPU:0',
        type=str,
        help='Device to use (/GPU:0 or /CPU:0)'
    )

    args = parser.parse_args()

    # Load images from input directory
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all .tif files sorted by name
    image_files = sorted(input_dir.glob('*.tif'))
    
    if not image_files:
        raise ValueError(f"No .tif files found in {input_dir}")
    
    # Already padded images
    print(f"Loading {len(image_files)} images from {input_dir}")
    images = [tifffile.imread(img_file) for img_file in image_files]
    images = np.array(images)
       
    print(f"Running DeepD3 inference on device {args.device}")
    
    # Run inference
    spine_predictions, dendrite_predictions = inference(
        images=images,
        model_fn=args.model,
        device=args.device
    )
    
    # Save predictions as .tif files
    for i, img_file in enumerate(image_files):
        spine_pred = spine_predictions[i]
        dendrite_pred = dendrite_predictions[i]

        spine_output_path = output_dir / "spines" / f"{img_file.stem}.tif"
        dendrite_output_path = output_dir / "dendrites" / f"{img_file.stem}.tif"

        spine_output_path.parent.mkdir(parents=True, exist_ok=True)
        dendrite_output_path.parent.mkdir(parents=True, exist_ok=True)

        tifffile.imwrite(spine_output_path, spine_pred.astype(np.float32))
        tifffile.imwrite(dendrite_output_path, dendrite_pred.astype(np.float32))



if __name__ == '__main__':
    main()
