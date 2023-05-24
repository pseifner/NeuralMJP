import gzip
import json
import os
import urllib.request
from functools import partial
from itertools import islice
from typing import Dict, List, Union
import io
import numpy as np

from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)


class _DownloadProgressBar(std_tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_file: str, filetype: str = '.txt.gz') -> None:
    with _DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        file, _ = urllib.request.urlretrieve(url, reporthook=t.update_to)

    if filetype == '.txt':
        os.rename(file, output_file)
    if filetype == '.txt.gz' or filetype == '.gz':
        batch_size: int = 1024 ** 2  # 1MB batches
        # Approximate progress bar since unzipped file size is unknown.
        with tqdm(total=os.path.getsize(file) // batch_size * 1.5, desc='Unpacking binary file', unit='MB') as pbar:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                with gzip.open(file, 'rb') as f_in:
                    while True:
                        file_content = f_in.read(batch_size).decode('utf-8', errors='ignore')
                        if not file_content:
                            break
                        f_out.write(file_content)
                        pbar.update(1)


def load_csv(file_path: str, num_of_lines: int = None):
    with io.open(file_path, 'r') as f:
        if num_of_lines is not None:
            data = np.loadtxt(islice(f, num_of_lines), delimiter=',')
        else:
            data = np.loadtxt(f, delimiter=',')
        return data
