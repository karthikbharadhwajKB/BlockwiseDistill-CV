# setup.py
from setuptools import setup, find_packages

setup(
    name="blockwise-distill",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "wandb",
        "pyyaml"
    ]
)
