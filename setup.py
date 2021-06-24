from setuptools import setup, find_packages

setup(
    name='ncsn',
    version='0.0.1',
    description='noise conditional score network',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
