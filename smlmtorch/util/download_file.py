import requests
from tqdm import tqdm

def download_file_pbar(url, filename):
    """
    Download a file with a progress bar.
    
    Args:
        url (str): Direct download URL
        filename (str): Name to save the file as
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as progress_bar:
        for data in response.iter_content(chunk_size=1024*64):
            size = file.write(data)
            progress_bar.update(size)

