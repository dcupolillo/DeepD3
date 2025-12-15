from pathlib import Path
import numpy as np
import flammkuchen as fl


def suggest_min_content(
        data_path: str or Path,
        patch_size: int = 128,
        num_samples: int = 10000,
        percentile: int = 10,
        min_threshold: int = 5
) -> int:
    """
    Suggest optimal min_content threshold by analyzing patch label distribution.
    
    Samples random patches from the dataset and calculates the percentile of labeled
    pixels. The recommended threshold ensures that the specified percentile of patches
    meet the minimum content requirement.

    Assumes that the data at `data_path` is structured as a DeepD3 .d3set file,
    containing 'dendrites' and 'spines' datasets and that each image has shape (1, H, W)
    and that labels are boolean or binary arrays.
    
    Parameters
    ----------
    data_path : str or Path
        Path to the .d3set file
    patch_size : int, optional
        Size of square patches to sample (default: 128)
    num_samples : int, optional
        Number of random patches to sample (default: 10000)
    percentile : int, optional
        Percentile to use for recommendation (default: 10, meaning 90% of patches
        will have at least this many labeled pixels)
    min_threshold : int, optional
        Minimum threshold value to return (default: 5)
        
    Returns
    -------
    int
        Recommended min_content value
    """
    data = fl.load(data_path)
    
    dendrite_counts = np.zeros(num_samples, dtype=int)
    spine_counts = np.zeros(num_samples, dtype=int)
    
    dendrites = data['data']['dendrites']
    spines = data['data']['spines']
    
    for i in range(num_samples):
        
        # Pick random image
        n = np.random.randint(0, len(dendrites))
        dendrite = dendrites[f"x{n}"]
        spine = spines[f"x{n}"]
        
        # Get image dimensions (assuming shape is (1, H, W))
        _, h, w = dendrite.shape
        
        # Skip if image is too small, which should not
        if h < patch_size or w < patch_size:
            continue
        
        # Random patch location
        y = np.random.randint(0, h - patch_size + 1)
        x = np.random.randint(0, w - patch_size + 1)
        
        # Extract patch
        dendrite_patch = dendrite[0, y:y+patch_size, x:x+patch_size]
        spine_patch = spine[0, y:y+patch_size, x:x+patch_size]
        
        # Count labeled pixels
        dendrite_counts[i] = np.sum(dendrite_patch)
        spine_counts[i] = np.sum(spine_patch)
    
    spine_counts = np.array(spine_counts)
    
    # Use specified percentile based on spines only
    recommended = max(
        np.percentile(spine_counts, percentile),
        min_threshold
    )
    
    return int(recommended)
