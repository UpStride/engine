import setuptools

with open('version') as f:
  version = f.readline()

with open("README.md", "r") as f:
  long_description = f.read()

setuptools.setup(
    name="upstride",
    version=version,
    author="UpStride S.A.S",
    author_email="hello@upstride.io",
    description="Upstride computation engine for Geometrical Algebra neural network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://upstride.io",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
