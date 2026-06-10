from setuptools import setup, find_packages

setup(
    name="vision_benchmark",
    version="0.1.0",
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "vision-bench=benchmark:cli",
        ]
    },
)
