from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        Extension("_lda", ["_lda.pyx", "gamma.c"]),
        Extension("_btm", ["_btm.pyx", "gamma.c"]),
        Extension("_jst", ["_jst.pyx", "gamma.c"]),
        Extension("_bjst", ["_bjst.pyx", "gamma.c"]),
    ]
)
