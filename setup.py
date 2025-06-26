"""
Setup script for OpenControl: Advanced Multi-Modal World Model Platform.

Author: Nik Jois <nikjois@llamasearch.ai>
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="opencontrol",
    version="1.0.0",
    author="Nik Jois",
    author_email="nikjois@llamasearch.ai",
    description="Advanced Multi-Modal World Model Platform for Embodied AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/llamasearchai/OpenControl",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Robotics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0", 
            "pytest-benchmark>=4.0.0",
            "pytest-cov>=4.1.0",
            "pytest-xdist>=3.3.0",
            "black>=23.9.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.6.0",
            "pre-commit>=3.5.0",
        ],
        "docs": [
            "sphinx>=7.2.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
        "monitoring": [
            "wandb>=0.16.0",
            "tensorboard>=2.15.0",
        ],
        "visualization": [
            "plotly>=5.17.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "opencontrol=opencontrol.cli.main:cli",
        ],
    },
    include_package_data=True,
    package_data={
        "opencontrol": [
            "configs/**/*.yaml",
            "configs/**/*.yml",
        ],
    },
    zip_safe=False,
    keywords=[
        "machine learning",
        "deep learning", 
        "robotics",
        "world model",
        "transformer",
        "multi-modal",
        "model predictive control",
        "embodied ai",
        "pytorch"
    ],
    project_urls={
        "Bug Reports": "https://github.com/llamasearchai/OpenControl/issues",
        "Source": "https://github.com/llamasearchai/OpenControl",
        "Documentation": "https://opencontrol.readthedocs.io/",
    },
) 