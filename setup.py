from setuptools import setup, Extension
import pybind11
import numpy

ext = Extension(
    "router",
    sources=["src/router_interval_space.cpp"],
    include_dirs=[
        pybind11.get_include(),
        numpy.get_include()
    ],
    extra_compile_args=["-O3", "-fopenmp", "-std=c++17"],
    extra_link_args=["-fopenmp"],
    language="c++"
)

setup(
    name="bitpacked_phase_router",
    version="0.1.0",
    packages=["src"],
    package_dir={"": "."},
    py_modules=["router_py"],
    ext_modules=[ext],
)