from __future__ import print_function, division
from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import os, sys

# Must have Cython installed, or it's a no-go
try:
    from Cython.Distutils import build_ext
except:
    print("You don't seem to have Cython installed. Please get a ")
    print("copy from www.cython.org and install it")
    sys.exit(1)


dirname = 'hwt'
packages = ['hwt', 'hwt.plot', 'hwt.cfuncs']
setup_path = os.path.split(os.path.abspath(__file__))[0]
sys.path.append(os.path.join(setup_path, dirname))
import version
ver = version.get_version()
sys.path.pop()


# Scan directory for extension files, converting
# them to extension names in dotted notation
def scandir(dir, files=[]):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".pyx"):
            files.append(path.replace(os.path.sep, ".")[:-4])
        elif os.path.isdir(path):
            scandir(path, files)
    return files

# Scan directory for extension files, removing compiled
# or html files
def cleanup(dir, fext):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(fext):
            os.remove(path)
        elif os.path.isdir(path):
            cleanup(path, fext)


# Generate an Extension object from its dotted name
def makeExtension(extName):
    extPath = extName.replace(".", os.path.sep)+".pyx"
    return Extension(
        extName,
        [extPath],
        include_dirs = [np.get_include(), "."],   # the '.' is CRUCIAL!!
        extra_compile_args = ["-O3", "-Wall"],
        extra_link_args = [],
        libraries = [],
        )


# get the list of extensions
extNames = scandir(dirname)

# and build up the set of Extension objects
extensions = [makeExtension(name) for name in extNames]

setup(name = 'HWT',
      version = ver,
      description       = 'Python Module for HWT Post Processing',
      author            = 'Patrick Marsh',
      author_email      = 'patrick.marsh@noaa.gov',
      url               = '',
      download_url      = '',
      packages          = packages,
      scripts           = [],
      cmdclass          = {'build_ext': build_ext},
      ext_modules       = extensions
      )

cleanup(dirname, '.c')
cleanup(dirname, '.html')
cleanup(dirname, '.pyc')
cleanup(dirname, '.pyd')
if sys.argv[1] not in ['build']:
    import shutil
    shutil.rmtree('build')
