from setuptools import setup, Extension, find_packages, Command
from setuptools.command.build_py import build_py as build_py_orig

from distutils.sysconfig import get_config_vars as default_get_config_vars
import distutils.sysconfig as dsc

from Cython.Build import cythonize
from Cython.Compiler import Options

# remove python doc strings in the so files 
Options.docstrings = False
Options.emit_code_comments = False


extensions = [
    Extension("upstride.convolutional", ["upstride/convolutional.py"]),
    Extension("upstride.generic_layers", ["upstride/generic_layers.py"]),
    Extension("upstride.type1.tf.keras.layers", ["upstride/type1/tf/keras/layers.py"]),
    Extension("upstride.type2.tf.keras.layers", ["upstride/type2/tf/keras/layers.py"]),
    Extension("upstride.type2.tf.keras.convolutional", ["upstride/type2/tf/keras/convolutional.py"]),
    Extension("upstride.type2.tf.keras.dense", ["upstride/type2/tf/keras/dense.py"]),
    Extension("upstride.type2.tf.keras.initializers", ["upstride/type2/tf/keras/initializers.py"]),
    Extension("upstride.type2.tf.keras.utils", ["upstride/type2/tf/keras/utils.py"]),
    Extension("upstride.type3.tf.keras.layers", ["upstride/type3/tf/keras/layers.py"]),
    Extension("upstride.type_generic.tf.keras.layers", ["upstride/type_generic/tf/keras/layers.py"])
]


# This is required so that .py files are not added to the build folder
class build_py(build_py_orig):
    def build_packages(self):
        pass


def remove_certain_flags(x):
    if type(x) is str:
        x = x.strip()
        x = x.replace("-DNDEBUG", "")
        x = x.replace("-g "," ")
        x = x.replace("-fstack-protector-strong","")
        # on the tensorflow 2.2 docker the compiler flag already has -fvisibility=hidden
    return x


def my_get_config_vars(*args):
  result = default_get_config_vars(*args)
  # sometimes result is a list and sometimes a dict:
  if type(result) is list:
     return [remove_certain_flags(x) for x in result]
  elif type(result) is dict:
     return {k : remove_certain_flags(x) for k,x in result.items()}
  else:
     raise Exception("cannot handle type"+type(result))

# replace original get_config_vars to the updated one.    
dsc.get_config_vars = my_get_config_vars

     
setup(
    name="upstride",
    version="1.0-cython",
    author="UpStride S.A.S",
    author_email="hello@upstride.io",
    description="A package to benefit from Upstride's Datatype.",
    url="https://upstride.io",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    ext_modules=cythonize(extensions,language_level=3,nthreads=12),
    cmdclass = {'build_py': build_py },
)
