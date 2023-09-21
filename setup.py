from setuptools import setup


setup(
    name="mlbench",
    packages=["mlbench"],
    description="ML metrics / benchmarking",
    license="MIT",
    author="Amir Ziai",
    author_email="arziai@gmail.com",
    url="https://github.com/amirziai/mlbench",
    keywords=["ml", "benchmarking", "metrics"],
    classifiers=[],
    install_requires=[
        "scipy",
        "numpy",
        "scikit-learn",
        "pandas",
        "matplotlib",
    ],
)
