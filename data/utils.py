import os
import os.path as osp
import requests
from tqdm import tqdm

def download_from_url(url, dst):
    """Download file

    Args:
        url (str): url to download file
        dst (str): place to put the file
    """
    file_size = int(requests.head(url).headers['Content-Length'])

    if osp.exists(dst):
        first_byte = osp.getsize(dst)
    else:
        first_byte = 0

    if first_byte >= file_size:
        return file_size

    header = {"Range": "bytes=%s-%s" % (first_byte, file_size)}

    pbar = tqdm(
        total=file_size, initial=first_byte,
        unit='B', unit_scale=True, desc=url.split('/')[-1])

    req = requests.get(url, headers=header, stream=True)
    with(open(dst, 'ab')) as f:
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                pbar.update(1024)
    pbar.close()
    return file_size

def download_from_dropbox(url, dst):
    """Download file from dropbox

    Args:
        url (str): url to download file
        dst (str): place to put the file
    """
    downloaded_file = requests.get(url)
    with open(dst, 'wb') as f:
        f.write(downloaded_file.content)
