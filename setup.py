from setuptools import setup

setup(
    name="es_distributed",
    version="0.1.0-dev",
    packages=["es_distributed"],
    install_requires=[
        "redis",
        "h5py",
        "tensorflow",
        "psutil",
        "gym",
        "click"
    ],
)
