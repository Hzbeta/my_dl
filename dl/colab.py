import os
import requests
import toml
from dl import file


def is_in_colab() -> bool:
    if os.getcwd() == '/content':
        return True
    else:
        return False

def init():
    nvidia_smi = "".join(os.popen('nvidia-smi').readlines())
    print(nvidia_smi)
    from google.colab import drive  # type: ignore 忽略pylance错误
    drive.mount('/content/drive')