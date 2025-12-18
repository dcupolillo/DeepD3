"""
Simple inference API for DeepD3 model predictions.

Created on Mon Oct 23 12:23:47 2023
@author: dcupolillo
"""

import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from deepd3.inference.utils import unpad_predictions


def inference(
        images: np.ndarray,
        model_fn: str or Path,
        original_dimensions: tuple or list = None,
        device: str = '/cpu:0',
) -> tuple:
    """
    Perform neural network inference on input images to obtain pixel-wise segmentation.

    Handles both single 2D images and batches of 2D images. Input is normalized to
    [-1, 1] range and fed to the model. Predictions are optionally unpadded to
    restore original dimensions.

    Parameters
    ----------
    images : np.ndarray
        Input images for segmentation:
        - Single image: 2D array (H, W)
        - Batch of images: 3D array (B, H, W)
        Images should be padded to match model input size if needed.
    model_fn : str or Path
        Path to the pre-trained TensorFlow model file (.h5).
    original_dimensions : tuple or list, optional
        Original image dimensions (H, W) before padding. If provided, predictions
        are cropped to these dimensions. If None, returns full predictions.
    device : str, optional
        Device for inference. Default: '/cpu:0'. Use '/gpu:0' for GPU inference.

    Returns
    -------
    tuple
        (spine_predictions, dendrite_predictions) as numpy arrays

    Notes
    -----
    - Images are normalized to [-1, 1] range before inference (same as training)
    - Model must have two outputs: dendrites and spines
    - Predictions are squeezed to remove batch dimension if single image input
    """

    # Add batch dimension if single image
    if images.ndim == 2:
        images = images[np.newaxis, ...]

    # Normalize to [-1, 1] range (matches training normalization)
    images_min = images.min(axis=(1, 2), keepdims=True)
    images_max = images.max(axis=(1, 2), keepdims=True)
    images = (images - images_min) / (images_max - images_min) * 2 - 1

    # Load model and run inference
    model_path = Path(model_fn)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with tf.device(device):
        model = load_model(model_path, compile=False)
        dendrite_pred, spine_pred = model.predict(images[..., None])

    # Remove batch/channel dimensions
    dendrite_pred = dendrite_pred.squeeze()
    spine_pred = spine_pred.squeeze()

    # Optionally unpad to original dimensions
    if original_dimensions is not None:
        dendrite_pred = unpad_predictions(dendrite_pred, original_dimensions)
        spine_pred = unpad_predictions(spine_pred, original_dimensions)

    return spine_pred, dendrite_pred
