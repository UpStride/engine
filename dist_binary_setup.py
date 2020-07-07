from setuptools import setup, Extension, find_packages 
from setuptools.command.build_py import build_py as build_py_orig

from distutils.sysconfig import get_config_vars as default_get_config_vars
import distutils.sysconfig as dsc

from Cython.Build import cythonize
from Cython.Compiler import Options

import glob

#remove python doc strings in the so files 
Options.docstrings = False
Options.emit_code_comments = False

compile_args = ["-fvisibility=protected",
                "-ffast-math", 
                "-funroll-loops", 
                "-floop-nest-optimize", 
                "-fipa-pta", 
                "-flto", 
                "-ftree-vectorize", 
                "-fno-signed-zeros", 
                "-march=native", 
                "-mfma"
            ]

py_files = glob.glob('upstride/**/*[!_].py',recursive=True)
ext_names = [x.split('.')[0].replace('/','.') for x in py_files]
ext_modules_list = list()
for name, pyfile in zip(ext_names, py_files):
    ext_modules_list.append(Extension(name,[pyfile],extra_compile_args=compile_args)) 


# This is required so that .py files are not added to the build folder
class build_py(build_py_orig):
    def build_packages(self):
        pass


def remove_certain_flags(x):
    if type(x) is str:
        x = x.strip()
        x = x.replace("-DNDEBUG", "")
        x = x.replace("-g "," ")
        x = x.replace("-D_FORTIFY_SOURCE=2","-D_FORTIFY_SOURCE=1")
        x = x.replace("-O1","-O3")
        x = x.replace("-O2","-O3")
        x = x.replace("-fstack-protector-strong","")
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
    version="1.0",
    author="UpStride S.A.S",
    author_email="hello@upstride.io",
    description="A package to benefit from Upstride's Datatype.",
    url="https://upstride.io",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    ext_modules=cythonize(ext_modules_list,language_level=3,nthreads=12),
    cmdclass = {'build_py': build_py },
)
