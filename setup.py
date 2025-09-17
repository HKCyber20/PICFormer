from setuptools import setup, find_packages

setup(
    name="picformer_project",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.0",
        "torchvision",
        "pyyaml",
        "numpy",
        "tqdm",
        "einops",
        "timm",
        "matplotlib==3.7.1",
        "scikit-image",
        "opencv-python",
        "ipython",
    ],
) 