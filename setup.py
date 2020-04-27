import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="upstride_python",
    version="0.1.0",
    author="UpStride S.A.S",
    author_email="hello@upstride.io",
    description="A package to use Geometrical Algebra in pure tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://upstride.io",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    test_suite="nose.collector",
    tests_require=['nose'],
    install_requires=[
        'PyYAML',
    ],
)
