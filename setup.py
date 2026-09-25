from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
import os
import subprocess
import sys


class CBuildExt(build_ext):
    def run(self):
        if sys.platform == "darwin":
            libname = "libtree.dylib"
            shared_flag = "-dynamiclib"
        elif sys.platform == "win32":
            libname = "tree.dll"
            shared_flag = "-shared"
        else:
            libname = "libtree.so"
            shared_flag = "-shared"

        src = os.path.join("smallgbm", "_c", "tree.c")
        out = os.path.join("smallgbm", "_c", libname)

        if not os.path.exists(src):
            print(f"[smallgbm] warning: {src} not found, skipping build")
            return super().run()

        cmd = ["cc", "-O3", "-fPIC", shared_flag, src, "-o", out, "-lm"]
        print("[smallgbm] building:", " ".join(cmd))
        subprocess.check_call(cmd)

        super().run()


setup(
    name="smallgbm",
    version="1.5.0",
    author="Emelyanov Ilya",
    author_email="Nsdmlk@yandex.ru",
    description="Gradient boosting optimized for small datasets (C backend)",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    package_data={
        "smallgbm": ["_c/*.c", "_c/*.h", "_c/*.dylib", "_c/*.so", "_c/*.dll"],
    },
    include_package_data=True,
    install_requires=["numpy>=1.20.0", "scikit-learn>=1.0.0"],
    python_requires=">=3.8",
    cmdclass={"build_ext": CBuildExt},
)