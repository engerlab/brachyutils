from setuptools import find_packages, setup

core_packages = [
    "setuptools",
    "cycler",
    "fonttools",
    "importlib-resources",
    "kiwisolver",
    "pybind11",
    "numpy",
    "matplotlib",
    "packaging",
    "pandas",
    "Pillow",
    "pyparsing",
    "python-dateutil",
    "pytz",
    "cmake",
    "scikit-build",
    "SimpleITK",
    "pynrrd @ git+https://github.com/mhe/pynrrd.git",
    "six",
    "tzdata",
    "zipp",
    "pyzstd",
    "typer",
    "tqdm",
    "DicomRTTool",
    "pydicom",
    "scipy",
    "tk",
    "pymedphys",
    "py7zr",
    "pytest",
    "vtk",
    # comment out for docker {
    "opentps @ git+https://github.com/engerlab/OpenTPS-brachyutils.git",
    "ai_assisted_brachy @ git+https://github.com/engerlab/AI_Assisted_Brachytherapy.git",
    # }
    "nibabel",
    "pydantic",
]

reg_extra = [
    "monai",
]

plan_extra = [
    "gurobipy",
    "amplpy",
    "ortools",
]

setup(
    name="brachyutils",
    version="0.3",
    description="Python utility packages for handling dose files and egsphant files.",
    author="EngerLab",
    packages=find_packages(include=["brachyutils.*", "brachyutils"]),
    package_dir={"": "."},  # required for pip install -e .
    install_requires=core_packages,
    extras_require={
        "core": core_packages,
        "reg": core_packages + reg_extra,
        "plan": core_packages + plan_extra,
        "full": core_packages + reg_extra + plan_extra,
    },
    entry_points={"console_scripts": ["brachyutils=brachyutils.cli_utils:main"]},
)
