import glob
import os
import os.path as osp

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

this_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join("pointnet2_ops", "_ext-src")
_ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
    osp.join(_ext_src_root, "src", "*.cu")
)
_ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

requirements = ["torch>=1.4"]

exec(open(osp.join("pointnet2_ops", "_version.py")).read())

# Which GPU architectures to emit code for.
#
# This used to be hardcoded to "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5" and assigned
# unconditionally, which breaks on any modern toolkit: CUDA 12 removed Kepler support,
# so nvcc rejects compute_37 outright with
#
#     nvcc fatal : Unsupported gpu architecture 'compute_37'
#
# and the build dies before producing anything. Because the assignment overwrote the
# environment variable, exporting TORCH_CUDA_ARCH_LIST beforehand could not work around
# it either.
#
# Now: an explicit TORCH_CUDA_ARCH_LIST always wins. Otherwise pick a default from the
# detected CUDA version, since the right answer differs per toolkit. Building only for
# the architectures you need is also several times faster than building for all of them.
if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
    try:
        from torch.utils.cpp_extension import CUDA_HOME
        import torch

        cuda_version = torch.version.cuda or ""
        major = int(cuda_version.split(".")[0]) if cuda_version else 0
    except Exception:
        major = 0

    if major >= 12:
        # Kepler (3.x) removed; Maxwell (5.x) deprecated. Pascal and newer only.
        # 6.0/6.1 = P100/T4-era, 7.0 = V100, 7.5 = T4, 8.0 = A100, 8.6 = A10/3090,
        # 9.0 = H100. +PTX on the last one keeps it forward-compatible with newer cards.
        arch_list = "6.0;6.1;7.0;7.5;8.0;8.6;9.0+PTX"
    elif major == 11:
        arch_list = "5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
    else:
        arch_list = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"      # the original list

    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list
    print(f"[pointnet2_ops] CUDA {cuda_version or 'unknown'} -> "
          f"TORCH_CUDA_ARCH_LIST={arch_list}")
else:
    print(f"[pointnet2_ops] using TORCH_CUDA_ARCH_LIST from the environment: "
          f"{os.environ['TORCH_CUDA_ARCH_LIST']}")
setup(
    name="pointnet2_ops",
    version=__version__,
    author="Erik Wijmans",
    packages=find_packages(),
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="pointnet2_ops._ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-Xfatbin", "-compress-all"],
            },
            include_dirs=[osp.join(this_dir, _ext_src_root, "include")],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    include_package_data=True,
)
