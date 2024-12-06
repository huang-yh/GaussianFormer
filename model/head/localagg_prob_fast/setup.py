#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="local_aggregate_prob_fast",
    packages=['local_aggregate_prob_fast'],
    ext_modules=[
        CUDAExtension(
            name="local_aggregate_prob_fast._C",
            sources=[
            "src/aggregator_impl.cu",
            "src/forward.cu",
            "src/backward.cu",
            "local_aggregate.cu",
            "ext.cpp"],
            # extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            # extra_compile_args={"nvcc": ["-g", "-G", "-Xcompiler", "-fno-gnu-unique","-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            # extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique","-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
            extra_compile_args={"nvcc": ["-Xcompiler", "-fno-gnu-unique"]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
