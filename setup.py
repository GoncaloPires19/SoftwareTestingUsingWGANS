from setuptools import setup, find_packages

setup(
    name="software-testing-wgans",
    version="1.0.0",
    description="Software Testing Using WGAN-GP for Synthetic JUnit Test Generation",
    author="Goncalo Pires",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch==2.0.1",
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "tqdm>=4.64.0",
        "scikit-learn>=1.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)
