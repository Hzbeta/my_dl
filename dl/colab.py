import os
import requests
import toml
from dl import file


def is_in_colab() -> bool:
    if os.getcwd() == '/content':
        return True
    else:
        return False

def config(gist_hash):
    conf=file.get_url_content(f"https://gist.githubusercontent.com/Hzbeta/{gist_hash}/raw/colab.toml")
    return toml.loads(conf)