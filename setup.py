import setuptools
import codecs

with codecs.open("README.md", encoding='utf-8') as fh:
  long_description = fh.read()

setuptools.setup(
    name="upstride",
    version="1.1.0",
    author="UpStride S.A.S",
    author_email="hello@upstride.io",
    description="A package to use Geometrical Algebra in pure tensorflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://upstride.io",
    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={'': ['_upstride_ops.so']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    # test_suite="",
    # tests_require=[''],
    install_requires=['pydot','graphviz']
)
