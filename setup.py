import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="celldancer",
    version="0.0.1",
    author="Lingqun Ye",
    author_email="yelingqun@gmail.com",
    description="Study RNA velocity through neural network.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={'': ['nn_pretrain_model/*.pt']},
    include_package_data=True,
    install_requires=[
        'pandas>=1.3.1',
        'numpy>=1.19.5',
        'scipy>=1.4.1',
        'scikit-learn>=0.22.1',
        'torch>=1.9.0',
        'pytorch-lightning>=1.4.4',
        'matplotlib>=3.1.3',
        'seaborn>=0.10.0',
        'joblib>=0.14.1'],
    python_requires=">=3.7",
)