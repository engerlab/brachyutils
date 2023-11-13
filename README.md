# BrachyUtils

This package implements Brachytherapy dose, egsphant dicom and film dosimetry functionalities. 

## Installation

To get the package run:

`git clone https://gitlab.com/hosseinjafar/tg186-validation.git`

Then, create a virtual envionrment and activate it by running:

`python3 -m venv ENV_brachyutils`

`source ENV_brachyutils/bin/activate`

`python3 -m pip install --upgrade pip`

Install SimpleITK independently by running `python3 -m pip install SimpleITK`. If you run into the error saying `skbuild` is [missing](https://bugs.python.org/issue30573), run `python3 -m pip install cmake`, then try installing SimpleITK again.

After this process finishes, run `pip install -e .` to install the brachyutils package. 

## brachyutils commands

brachyutils comes with a linux command line interface, to learn about its functionality run:

`brachyutils --help` on the command line. 

## BrachyDose

You can import this object in your python script by running `from brachyutils import BrachyEgsphant`. This object has the following attributes and functions:

```to be added later```

## BrachyEgsphant

You can import this object in your python script by running `from brachyutils import BrachyDose`. This object has the following attributes and functions:

```to be added later```

We use DicomRTTools to extract info from dicom files. For more info on this package, please visit the [DicomRTTools paper and repository](https://www.sciencedirect.com/science/article/abs/pii/S1879850021000485)

