# Zerodose

![Alt text](resources/imgs/banner.jpg)

A tool to assist in personalized abnormality investigation in combined FDG-PET/MRI imaging.
Created by the department of [Clinically Applied Artificial Intelligence](http://caai.dk/) at [Copenhagen University Hospital](https://www.rigshospitalet.dk/)

[![PyPI](https://img.shields.io/pypi/v/zerodose.svg)][pypi_]
[![Status](https://img.shields.io/pypi/status/zerodose.svg)][status]
[![Python Version](https://img.shields.io/pypi/pyversions/zerodose)][python version]
[![License](https://img.shields.io/pypi/l/zerodose)][license]
[![Read the documentation at https://zerodose.readthedocs.io/](https://img.shields.io/readthedocs/zerodose/latest.svg?label=Read%20the%20Docs)][read the docs]
[![Tests](https://github.com/ChristianHinge/zerodose/workflows/Tests/badge.svg)][tests]

[pypi_]: https://pypi.org/project/zerodose/
[status]: https://pypi.org/project/zerodose/
[read the docs]: https://zerodose.readthedocs.io/
[tests]: https://github.com/ChristianHinge/zerodose/actions?workflow=Tests
[python version]: https://pypi.org/project/zerodose

## Installation

Note that a python3 installation is required for _Zerodose_ to work.
You can install _Zerodose_ via [pip] from [PyPI]:

```console
$ pip install zerodose
```

## Usage
Note!: 
- All input images should be affinely registered to [MNI 2009a Nonlinear Symmetric/Assymmetric](https://nist.mni.mcgill.ca/icbm-152-nonlinear-atlases-2009/) space (1×1x1mm). Use `zerodose pipeline` if ZeroDose should do the registration. 
- A brain mask is required to run ZeroDose - we can recommend [HD-BET](https://github.com/MIC-DKFZ/HD-BET). 

### Run zerodose

Create an sbPET and abnormality map:
```console
$ zerodose run -i mr.nii.gz -p pet.nii.gz -m brain_mask.nii.gz -os sb_pet.nii.gz -oa abn.nii.gz
```

### Run pipeline
Identical to `zerodose run` but with registration ([NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg)) to and from MNI space. (Registration may take several minutes depending on image dimensions)
```console
$ zerodose pipeline -i mr.nii.gz -p pet.nii.gz -m brain_mask.nii.gz -os sb_pet.nii.gz -oa abn.nii.gz
```

### Run individual steps
#### Synthesize raw baseline PET

```console
$ zerodose syn -i mr.nii.gz -m brain_mask.nii.gz -o sb_pet_raw.nii.gz
```
#### Intensity normalize raw sbPET

```console
$ zerodose norm -i mr.nii.gz -m brain_mask.nii.gz -o sb_pet_raw.nii.gz
```

#### Create abnormality map

```console
$ zerodose abn -p pet.nii.gz -s sb_pet.nii.gz -m brain_mask.nii.gz -o abn.nii.gz
```

Please see the [Command-line Reference] for details.

## Docker

- TODO

## Hardware requirements
- TODO

## Issues and contributing

Contributions are very welcome.
If you encounter any problems,
please [file an issue] along with a description.

[pypi]: https://pypi.org/
[file an issue]: https://github.com/ChristianHinge/zerodose/issues
[pip]: https://pip.pypa.io/

<!-- github-only -->

[license]: https://github.com/ChristianHinge/zerodose/blob/main/LICENSE
[contributor guide]: https://github.com/ChristianHinge/zerodose/blob/main/CONTRIBUTING.md
[command-line reference]: https://zerodose.readthedocs.io/en/latest/usage.html
