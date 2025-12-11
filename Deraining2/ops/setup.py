from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='irnn',
    ext_modules=[
        CUDAExtension(
            name='irnn',
            sources=['src/irnn.cpp', 'src/irnn_kernel.cu'],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-O3']
            }
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
