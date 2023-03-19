import setuptools


def read_install_requires():
    print("Reading install requirements")
    return [r.split('\n')[0] for r in open('requirements.txt', 'r').readlines()]


def get_log_description():
    print("Reading README File")
    with open("README.md", "r") as fh:
        long_description = fh.read()
    return long_description


setuptools.setup(
    name="pynncml",
    author="Hai Victor Habi",
    author_email="victoropensource@gmail.com",
    description="A python toolbox based on PyTorch which utilized neural network for rain estimation and classification from commercial microwave link (CMLs) data.",
    long_description=get_log_description(),
    long_description_content_type="text/markdown",
    install_requires=read_install_requires(),
    python_requires='>=3.6',
    url="https://github.com/haihabi/pynncml",
    packages=setuptools.find_packages(exclude=['tests']),
    package_data={'pynncml.model_zoo': ['*']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
