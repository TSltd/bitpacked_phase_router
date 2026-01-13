from setuptools import setup, Extension
import pybind11
import numpy

ext = Extension(
    "router",
    sources=["src/router.cpp"],  
    include_dirs=[
        pybind11.get_include(),
        numpy.get_include()
    ],
    extra_compile_args=["-O3", "-fopenmp", "-std=c++17"], 
    extra_link_args=["-fopenmp"],
    language="c++"
)

setup(
    name="router",
    ext_modules=[ext],
)
