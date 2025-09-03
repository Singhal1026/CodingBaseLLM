from setuptools import setup, find_packages

setup(
    name="coding_base_llm",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tqdm>=4.65.0",
    ]
)