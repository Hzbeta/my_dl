from dl import file
import unittest, os
import shutil
import toml
from os.path import exists as pexists
from os.path import join as pjoin


class TestFile(unittest.TestCase):

    def assertDir(self, correct_dir_struct: dict, base_path: str) -> bool:
        #检测该文件夹中的文件
        #文件数目相等
        self.assertEqual(len(list(os.walk(base_path))[0][2]),len(correct_dir_struct['files']))
        #文件内容相同
        for f, content in correct_dir_struct['files']:
            file_path = pjoin(base_path, f)
            self.assertTrue(pexists(file_path))
            with open(file_path, encoding='utf8') as f:
                self.assertEqual(f.read(),content)
        #检测该文件夹中的子文件夹
        #子文件夹数目相等
        self.assertEqual(len(list(os.walk(base_path))[0][1]),len(correct_dir_struct['dirs']))
        #子文件夹内容相同
        for subdir_name, correct_subdir_struct in correct_dir_struct['dirs']:
            self.assertDir(correct_subdir_struct, pjoin(base_path, subdir_name))
        return True

    def test_Dataset(self):

        #清理磁盘上的所有测试数据集
        conf_path='./tests/mock/config/basic.toml'
        with open(conf_path, encoding='utf8') as f:
            conf = toml.load(f)
        dataset_path = conf['local_dataset_path']
        cache_path = conf['local_dataset_cache_path']
        shutil.rmtree(dataset_path, ignore_errors=True)
        shutil.rmtree(cache_path, ignore_errors=True)

        #开始测试
        dataset = file.Dataset(conf_path)
        test_data_path = dataset.download_extract('test')

        #测试是否下载并解压成功
        self.assertTrue(pexists(dataset_path))
        self.assertTrue(pexists(cache_path))
        cache_zip_path = pjoin(cache_path, 'test.zip')
        self.assertTrue(pexists(cache_zip_path))

        #检查解压出来的文件是否正确
        #正确的文件列表和对应的内容
        correct_dir_struct = {
            'files': [('file1.txt', 'text1')],
            'dirs': [
                ('dir1', {
                    'files': [
                        ('file1.txt', 'text1'),
                        ('file2.txt', 'text2'),
                    ],
                    'dirs': [],
                }),
                ('dir2', {
                    'files': [],
                    'dirs': [],
                }),
            ],
        }
        self.assertDir(correct_dir_struct,test_data_path)

if __name__ == '__main__':
    unittest.main()