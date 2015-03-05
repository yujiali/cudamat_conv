#!/usr/bin/env python

import os
from distutils.spawn import spawn
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import sys

# CUDA specific config
# nvcc is assumed to be in user's PATH
nvcc_compile_args = ['-O', '--ptxas-options=-v', '--compiler-options',
                     "'-fPIC'"]
nvcc_compile_args = os.environ.get('NVCCFLAGS', '').split() + nvcc_compile_args
cuda_libs = ['cublas']

cudamat_ext = Extension('cudamat_conv.libcudamat',
                        sources=['cudamat_conv/cudamat.cu',
                                 'cudamat_conv/cudamat_kernels.cu'],
                        libraries=cuda_libs,
                        extra_compile_args=nvcc_compile_args)
cudalearn_ext = Extension('cudamat_conv.libcudalearn',
                          sources=['cudamat_conv/learn.cu',
                                   'cudamat_conv/learn_kernels.cu'],
                          libraries=cuda_libs,
                          extra_compile_args=nvcc_compile_args)


class CUDA_build_ext(build_ext):
    """
    Custom build_ext command that compiles CUDA files.
    Note that all extension source files will be processed with this compiler.
    """
    def build_extensions(self):
        self.compiler.src_extensions.append('.cu')
        self.compiler.set_executable('compiler_so', 'nvcc')
        self.compiler.set_executable('linker_so', 'nvcc --shared')
        self.compiler.spawn = self.spawn
        build_ext.build_extensions(self)

    def spawn(self, cmd, search_path=1, verbose=0, dry_run=0):
        """
        Perform any CUDA specific customizations before actually launching
        compile/link etc. commands.
        """
        if (sys.platform == 'darwin' and len(cmd) >= 2 and cmd[0] == 'nvcc' and
                cmd[1] == '--shared' and cmd.count('-arch') > 0):
            # Versions of distutils on OSX earlier than 2.7.9 inject
            # '-arch x86_64' which we need to strip while using nvcc for
            # linking
            while True:
                try:
                    index = cmd.index('-arch')
                    del cmd[index:index+2]
                except ValueError:
                    break
        spawn(cmd, search_path, verbose, dry_run)

setup(name="cudamat_conv",
      version="0.1",
      description="Performs linear algebra computation on the GPU via CUDA",
      ext_modules=[cudamat_ext, cudalearn_ext],
      packages=find_packages(exclude=['examples', 'test']),
      include_package_data=True,
      package_data={'cudamat_conv': ['rnd_multipliers_32bit.txt']},
      author="Volodymyr Mnih, Yujia Li",
      url="https://github.com/yujiali/cudamat_conv",
      cmdclass={'build_ext': CUDA_build_ext})
