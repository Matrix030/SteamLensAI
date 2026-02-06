#!/usr/bin/env python
"""Setup script for steamLensAI."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="steamLensAI",
    version="0.1.0",
    author="Your Name",
    description="Steam game review analysis and summarization tool",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/steamLensAI",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.22.0",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "psutil>=5.9.0",
        "dask>=2023.3.0",
        "distributed>=2023.3.0",
        "tqdm>=4.64.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "sentence-transformers>=2.2.2",
        "transformers>=4.28.0,<5.0.0",
        "scikit-learn>=1.2.0",
        "pyarrow>=12.0.0",
        "accelerate>=0.20.0",
        "bokeh>=3.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)
