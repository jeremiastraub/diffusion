from setuptools import setup, find_packages

setup(
    name='diffusion',
    version='0.0.1',
    description='Generative Diffusion Models and representation learning',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
    ],
)
