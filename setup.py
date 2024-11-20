from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

setup(
    name="ft_linear_regression",
    version="0.0.1",
    description="A simple linear regression model to predict car prices.",
    long_description=open('README.md').read(),
    author="Semiha Beyazkilic",
    author_email="seozcan@student.42.com",
    url="https://github.com/semx2a/ft_linear_regression",
    packages=find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS independent",
    ],
    python_requires=">=3.6, <4",
)
