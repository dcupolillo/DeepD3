import imageio.v2 as io
from pathlib import Path
import numpy as np

# Example of how to convert a 2D image to a `dummy` 3D z-stack with one plane,
# so that it can be processed by DeepD3 network.
# Thanks to @ankilab for the code snippet
# https://github.com/ankilab/DeepD3

def dummy_stack(
        bidimensional_image: np.ndarray,
        output_path: str or Path = None
) -> np.ndarray:
    """
    Convert a 2D image to a dummy 3D z-stack with one plane.
    This is useful for processing 2D images with models expecting 3D input.

    Parameters
    ----------
    bidimensional_image : np.ndarray
        A 2D array representing the input image.
    output_path : str or Path, optional
        If provided, the path where the dummy stack will be saved as a TIFF file.
        If None, the image will not be saved. Default is None.
    
    Returns
    -------
    np.ndarray
        A 3D array representing the dummy z-stack with one plane.
    """
    if bidimensional_image.ndim != 2:
        raise ValueError("Input image must be a 2D array")

    # Create a z-stack with 1 plane
    hacked_image = bidimensional_image[None, ...]

    if output_path is not None:
        io.mimwrite(output_path, hacked_image)
    
    return hacked_image
