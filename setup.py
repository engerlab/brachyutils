from setuptools import setup, find_packages

setup(name='brachyutils',
      version='1.0',
      description='Python utility packages for handling dose files and egsphant files.',
      author='EngerLab',
      packages = find_packages(),
      install_requires=[
        "contourpy",
        "cycler",
        "fonttools",
        "importlib-resources",
        "kiwisolver",
        "matplotlib",
        "numpy",
        "packaging",
        "pandas",
        "Pillow",
        "pyparsing",
        "python-dateutil",
        "pytz",
        "SimpleITK",
        "six",
        "tzdata",
        "zipp",
        "pyzstd",
        "typer",
        "tqdm",
        "DicomRTTool",
        "scikit-build"
      ],
      entry_points={'console_scripts': ['brachyutils=brachyutils:main']},
      package_dir={'':'brachyutils/src'}
     )