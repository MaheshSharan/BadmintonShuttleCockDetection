from setuptools import setup, find_packages

setup(
    name="shuttlecock_detection",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'opencv-python',
        'tensorboard',
        'tqdm',
        'matplotlib',
        'plotly',
        'dash'
    ]
)
