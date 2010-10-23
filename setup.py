import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


libdirs=[]
incdirs=[numpy.get_include()]
libraries=[]

setup(name = "HWT",
      version = "0.1",
      description       = "Python Module for HWT Post Processing",
      author            = "Patrick Marsh",
      author_email      = "patrick.marsh@noaa.gov",
      url               = "",
      download_url      = "",
      scripts = [],
      cmdclass = {'build_ext': build_ext},
      ext_modules = [Extension("hwt", ["src/hwt.pyx"],
                               include_dirs=incdirs,
                               library_dirs=libdirs,
                               libraries=libraries
                              )
                    ]
      )
