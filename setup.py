from setuptools import setup, find_packages

setup(
    name="PSF_python",
    version="0.1",
    description="Modified version of QuantStats",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "scipy",
    ],
)