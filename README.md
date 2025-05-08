# qLDPC Decoding

This repository contains code that explores decoding circuit level noise for [bivariate bicyle codes](https://arxiv.org/abs/2308.07915), as well as surface codes.

In order to run the code in this environment, create a conda environment with the following command (this will result in a cpu install of pytorch):

```bash
conda env create --f environment_cpu.yml
```
This creates a conda environment named `qLDPC-decoding`.

If you have access to a Linux machine with an Nvidia gpu (with support for cuda 11.8), you can create a conda environment with a cuda install of pytorch with
```bash
conda env create --f environment_gpu.yml
```

To run tests, simply run
```bash
pytest
```

from the root directory of the repository.

Training models is performed with the `train.py` script.
To see the available options for running this script, run `python train.py --help`.
In order to evaluate a saved model, make sure to save the std out from the `train.py` script to a file.
The name of this log file is used as an input for the `eval_model.py` script.
