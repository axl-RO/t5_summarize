from setuptools import setup, find_packages

setup(
    name="t5-summarizer",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch",
        "transformers",
        "sentencepiece",
    ],
    entry_points={
        "console_scripts": [
            "t5summarize = summarizer.cli:main"
        ]
    },
)
