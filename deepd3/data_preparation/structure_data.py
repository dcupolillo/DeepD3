
import numpy as np
from datetime import datetime
import flammkuchen as fl
import pandas as pd
from pathlib import Path


def create_d3data(
        image_fn: str or Path,
        image: np.ndarray,
        dendrite: np.ndarray,
        spines: np.ndarray,
        resolution: float,
        save_fn: str = "data.d3data"
) -> None:
    """
    Create a DeepD3-compatible .d3data file from image stack and labels.

    Parameters
    ----------
    image_fn : str or Path
        Path to the original image file.
    image : np.ndarray
        3D array representing the image stack.
    dendrite : np.ndarray
        3D array representing the dendrite labels.
    spines : np.ndarray
        3D array representing the spine labels.
    save_fn : str, optional
        Filename for the output .d3data file. Default is "data.d3data".

    Returns
    -------
    None
        Saves a .d3data file with the provided data and metadata.
    """

    if not save_fn.endswith('.d3data'):
        raise ValueError("Output filename should end with .d3data extension")
    
    if not (image.shape == dendrite.shape == spines.shape):
        raise ValueError("Stack, dendrite, and spines must have the same shape")

    x, y, w, h, z_begin, z_end = 0, 0, image.shape[1], image.shape[0], 0, 1 

    data = {
        'stack': image,
        'dendrite': dendrite > 0,
        'spines': spines > 0
    }

    # DeepD3-structured metadata dictionary
    meta = {
        'crop': False,
        'X': x,
        'Y': y,
        'Width': w,
        'Height': h,
        'Depth': z_end-z_begin+1,
        'Z_begin': z_begin,
        'Z_end': z_end,
        'Resolution_XY': resolution,
        'Resolution_Z': 1, # Dummy value, adjust as needed
        'Timestamp': datetime.now().strftime(r"%Y%m%d_%H%M%S"),
        'Generated_from': image_fn
    }

    fl.save(save_fn, dict(data=data, meta=meta), compression='blosc')

    print(
        f"Saved {save_fn} with stack shape {image.shape} "
        f"and resolution {resolution} µm/px")


def create_d3set(
        files_list: list,
        save_fn: str or Path = "set.d3set"
) -> None:
    """
    Create a DeepD3-compatible .d3set file from multiple .d3data files.

    Parameters
    ----------
    files_list : list of str
        List of paths to .d3data files to be included in the .d3set.
    save_fn : str
        Filename for the output .d3set file.

    Returns
    -------
    None
        Saves a .d3set file with the combined data and metadata.
    """

    if not all(fn.endswith('.d3data') for fn in files_list):
        raise ValueError("All input filenames should end with .d3data extension")
    
    if not save_fn.endswith('.d3set'):
        raise ValueError("Output filename should end with .d3set extension")

    stacks = {}
    dendrites = {}
    spines = {}
    meta = pd.DataFrame()

    # For each dataset, add to set
    for i in range(len(files_list)):
        fn = files_list[i]
        print(fn, "...")

        d = fl.load(fn)

        stacks[f"x{i}"] = d['data']['stack']
        dendrites[f"x{i}"] = d['data']['dendrite']
        spines[f"x{i}"] = d['data']['spines']

        m = pd.DataFrame([d['meta']])
        meta = pd.concat((meta, m), axis=0, ignore_index=True)     

    fl.save(
        save_fn,
        dict(data=dict(stacks=stacks, dendrites=dendrites, spines=spines),
        meta=meta),
        compression='blosc')
    
    print(f"Saved {save_fn} with {len(files_list)} datasets.")