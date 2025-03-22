from setuptools import setup, find_packages

setup(
    name="sysgen",  
    version="0.1.1",
    author="Adhishtanaka",
    author_email="kulasoooriyaa@gmail.com",
    description="sysgen is a CLI tool that creates high-quality synthetic datasets using the Gemini API",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/adhishtanaka/sysgen",  
    license="MIT",
    packages=find_packages(),
    install_requires=[
         "google-genai",
    ],
    entry_points={
        "console_scripts": [
            "sysgen=sysgen.cli:main",  
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
