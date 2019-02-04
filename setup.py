from setuptools import setup

setup(
    name="es_distributed",
    version="0.1.0-dev",
    packages=["es_distributed"],
    install_requires=[
        "redis",
        "h5py",
        "tensorflow==1.3.0",
        "psutil",
        "gym==0.10.5",
        "grpcio==1.2.1",
        "protobuf==3.3.0",
        "click"
    ],
)
