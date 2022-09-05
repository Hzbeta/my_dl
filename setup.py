from setuptools import setup, find_packages

requirements = [
    'fire',
    'matplotlib',
    'numpy',
    'opencv-python',
    'toml',
    'tqdm',
    'yapf',
    'torchsampler',
    'ipykernel',
    'ipywidgets',
    'torchkeras',
    'python-docx',
    'jupyter',
    'pandas',
    'requests',
]

setup(
    name='dl',
    version='0.0.1',
    python_requires='>=3.7',
    author='Hzbeta',
    author_email='ihzbeta@outlook.com',
    url='https://github.com/Hzbeta/my_dl',
    description='My deep learning python toolkit',
    packages=find_packages(),
    install_requires=requirements,
)