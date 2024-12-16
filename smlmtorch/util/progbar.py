
import numpy as np

USE_AUTO_TQDM = True

custom_progbar_hook = None

def progbar(*args, **kwargs):
    if custom_progbar_hook is not None:
        return custom_progbar_hook(*args, **kwargs)

    if USE_AUTO_TQDM:
        from tqdm.auto import tqdm
    else:
        from tqdm import tqdm

    return tqdm(*args, **kwargs)

def pb_range(*args, **kwargs):
    return progbar(np.arange(*args), **kwargs)

