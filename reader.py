import os

import requests


def read_files(url, file_dir):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(file_dir, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print('red')
