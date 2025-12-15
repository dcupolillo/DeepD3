import random
import numpy as np
import tensorflow as tf


def schedule(
        epoch: int,
        learning_rate: float,
) -> float:
    """
    Learning rate scheduler that decays the learning rate exponentially after 15 epochs.

    Parameters
    ----------
    epoch : int
        Current epoch number.
    learning_rate : float
        Initial learning rate.
    
    Returns
    -------
    float
        Adjusted learning rate.
    """

    if epoch < 15:
        return learning_rate
    else:
        return learning_rate * tf.math.exp(-0.1)


def set_seed(
        seed: int = None,
) -> None:
    """
    Set the random seed for reproducibility.

    Parameters
    ----------
    seed : int, optional
        Random seed value. If None, a random seed is chosen.
    """

    if seed is None:
        seed = np.random.choice(2 ** 32)
    
    random.seed(seed)
    np.random.seed(seed)
    
    print(f'Random seed {seed} has been set.')