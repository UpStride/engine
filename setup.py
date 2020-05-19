import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="upstride",
    version="0.1.3",
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
    # test_suite="",
    # tests_require=[''],
    # install_requires=[
    #     '',
    # ],
)
