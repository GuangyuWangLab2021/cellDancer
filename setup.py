import setuptools

project_urls = {
  'cellDancer': 'https://github.com/GuangyuWangLab2021/cellDancer',
  'Documentation':'https://guangyuwanglab2021.github.io/cellDancer_website/'
}

with open("readme_pypi.rst", "rt", encoding="utf8") as f:
    long_description = f.read()

setuptools.setup(
    name="celldancer",
    version="1.1.7",
    author="Wang Lab",
    author_email="gwang2@houstonmethodist.org",
    description="Study RNA velocity through neural network.",
    long_description=long_description,
    long_description_content_type="text/x-rst; charset=UTF-8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls = project_urls,
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data={'': ['model/*.pt']},
    include_package_data=True,
    python_requires=">=3.7.6",
    install_requires = ['pytorch-lightning==1.5.2',
                        'torch==1.10.0',
                        'pandas==1.3.4',
                        'numpy==1.20.3',
                        'anndata==0.8.0',
                        'tqdm==4.62.3',
                        'scikit-learn==1.0.1',
                        'scipy==1.7.2',
                        'joblib==1.1.0',
                        'scikit-image==0.19.2',
                        'statsmodels==0.13.1',
                        'matplotlib==3.5.3',
                        'seaborn==0.11.2',
                        'datashader==0.14.0',
                        'bezier==2021.2.12',
                        'umap-learn==0.5.2',
                        'jupyterlab',
                        'setuptools==59.5.0',
                        'setuptools-scm==6.3.2'
                        ]
)

