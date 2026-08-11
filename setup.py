from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="smallgbm",
    version="1.0.0",
    author="Emelyanov Ilya",
    author_email="Nsdmlk@yandex.ru",
    description="Gradient boosting optimized for small datasets (n < 1000)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nsdmlk/SmallGBM",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scikit-learn>=1.0.0",
    ],
)