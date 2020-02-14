import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="pynncml",
    version="0.1",
    author="Hai Victor Habi",
    author_email="victoropensource@gmail.com",
    description="A python toolbox based on PyTorch which utilized neural network for rain estimation and classification from commercial microwave link (CMLs) data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=required,
    url="https://github.com/haihabi/torch_rain",
    packages=setuptools.find_packages(exclude=['tests']),
    package_data={'pynncml.model_zoo':['*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
