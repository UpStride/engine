# Upstride Python Engine

Please have a look at https://upstride.atlassian.net/wiki/spaces/benchmarks/pages/942342147/Python+Engine+v2 For details regarding the current status of the python engine


# Versioning

The versioning here follow roughly the Semantic Versioning: 

- First number evolve when there is an API change and users need to be aware of it
- Second number evolve when new feature are introduce
- 3rd number evolve for bug fix, cleaning, small stuff

One exception: as this engine is focused on research, the notion of alpha and beta versions doesn't make much sense here.

# Testing

* To run the unittests, run `python test.py`
* To get the coverage, run `coverage run test.py`. Then `coverage report` show the coverage information and `coverage xml` create a file usable by VSCode
