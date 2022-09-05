'''
存放数据库初始化和文件相关的工具函数
'''
import os, hashlib, requests
import shutil
import uuid
import zipfile
import toml
import fire
from tqdm import tqdm


def get_vaild_filename(dir_path: str, file_ext: str) -> str:
    '''输入一个目录路径和后缀名，返回一个随机的合法的文件路径
    参数：
        dir_path:目录路径
        file_ext:文件后缀名
    返回：
        合法的文件路径
    '''
    assert os.path.exists(dir_path), '目录不存在'
    filename = str(uuid.uuid4())
    while (os.path.exists(os.path.join(dir_path, filename + file_ext))):
        filename = str(uuid.uuid4())
    return os.path.join(dir_path, filename + file_ext)


def get_file_sha1(file_path: str) -> str:
    '''输入文件路径，返回sha1值
    参数：
        file_path:文件路径
    返回：
        sha1值
    '''
    assert os.path.exists(file_path), '文件不存在'
    sha1 = hashlib.sha1()
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(1048576)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


def download_from_url(url, fname):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def get_url_content(url: str) -> str:
    req = requests.get(url)
    return req.content.decode('utf-8')

class Dataset():
    '''数据集文件类，用来下载和管理数据集'''

    def __init__(self, basic_config):
        '''需要配置文件的路径或解析后的字典'''
        if not isinstance(basic_config, dict):
            assert os.path.exists(basic_config), "配置文件不存在"
            with open(basic_config, encoding='utf8') as f:
                self.conf = toml.load(f)
        else:
            self.conf = basic_config
        try:
            self.dataset_path = self.conf['local_dataset_path']
            os.makedirs(self.dataset_path, exist_ok=True)
            get_url_content
            self.dataset_conf = toml.loads(get_url_content(self.conf['dataset_config']))
        except KeyError as e:
            raise Exception(f'基础配置文件中缺少键值：{e}')

    def download_extract(self, name: str, folder: str = None, re_extract: bool = False) -> str:
        """下载并解压zip/tar文件
            指定folder则返回指定子目录
        """
        try:
            url = self.dataset_conf[name]['url']
            sha1_hash = self.dataset_conf[name]['sha1_hash']
            cache_dir = self.conf['local_dataset_cache_path']
        except KeyError as e:
            raise Exception(f'数据库配置文件中缺少键值：{e}')
        os.makedirs(cache_dir, exist_ok=True)
        file_full_path = os.path.join(cache_dir, url.split('/')[-1])
        if os.path.exists(file_full_path) and get_file_sha1(file_full_path) == sha1_hash:
            pass  # 命中缓存
        else:
            print(f'正在从{url}下载到{file_full_path}...')
            download_from_url(url, file_full_path)
            re_extract = True  #需要重新下载的都需要重新解压
        #获得压缩包路径
        _, file_full_name = os.path.split(file_full_path)
        file_name, file_ext = os.path.splitext(file_full_name)
        #保存路径为数据集目录+压缩包名称
        extract_dir = os.path.join(self.dataset_path, file_name)
        #需要重新解压时，删除现有文件
        if re_extract and os.path.exists(extract_dir):
            shutil.rmtree(extract_dir, ignore_errors=True)
        #解压
        if not os.path.exists(extract_dir):
            if file_ext == '.zip':
                fp = zipfile.ZipFile(file_full_path, 'r')
            else:
                assert False, '只有zip文件可以被解压缩'
            print(f"正在解压到{extract_dir}...")
            for member in tqdm(fp.infolist(), desc='解压中'):
                fp.extract(member, extract_dir)
            # fp.extractall(extract_dir)
        return os.path.join(extract_dir, folder) if folder else extract_dir


if __name__ == "__main__":
    '''使工具函数可以直接由命令行调用'''
    fire.Fire()