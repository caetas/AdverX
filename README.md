# AdverX

[![Python](https://img.shields.io/badge/python-3.10+-informational.svg)](https://www.python.org/downloads/release/python-31014/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=black)](https://pycqa.github.io/isort)
[![documentation](https://img.shields.io/badge/docs-mkdocs%20material-blue.svg?style=flat)](https://mkdocstrings.github.io)
[![wandb](https://img.shields.io/badge/tracking-wandb-blue)](https://wandb.ai/site)
[![dvc](https://img.shields.io/badge/data-dvc-9cf)](https://dvc.org)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

Official repository for AdverX.

## Prerequisites

You will need:

- `python` (see `pyproject.toml` for full version)
- `Git`
- `Make`
- a `.secrets` file with the required secrets and credentials
- load environment variables from `.env`
- `CUDA >= 12.1`
- `Weights & Biases` account

## Installation

Clone this repository (requires git ssh keys)

    git clone --recursive git@github.com:caetas/AdverX.git
    cd adverx

Install conda environment

    conda env create -f environment.yml
    conda activate python3.10

### On Linux

And then setup all virtualenv using make file recipe

    (python3.10) $ make setup-all

You might be required to run the following command once to setup the automatic activation of the conda environment and the virtualenv:

    direnv allow

Feel free to edit the [`.envrc`](.envrc) file if you prefer to activate the environments manually.

### On Windows

You can setup the virtualenv by running the following commands:

    python -m venv .venv-dev
    .venv-dev/Scripts/Activate.ps1
    python -m pip install --upgrade pip setuptools
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    python -m pip install -r requirements/requirements-win.txt


To run the code please remember to always activate both environments:

    conda activate python3.10
    .venv-dev/Scripts/Activate.ps1

## Tracking

The code examples are setup to use [Weights & Biases](https://wandb.ai/home) as a tool to track your training runs. Please refer to the [`full documentation`](https://docs.wandb.ai/quickstart) if required or follow the following steps:

1. Create an account in [Weights & Biases](https://wandb.ai/home)
2. **If you have installed the requirements you can skip this step**. If not, activate the conda environment and the virtualenv and run:

    ```bash
    pip install wandb
    ```
3. Run the following command and insert you [`API key`](https://wandb.ai/authorize) when prompted:

    ```bash
    wandb login
    ```

## Dataset

AdverX utilizes the **first iteration** of the BIMCV-COVID-19+ dataset, that can be downloaded [`here`](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711). However, the dataset needs to be re-organized to a format that is compatible with the analysis performed in this work. **Please follow these steps**.

1. Download the **first iteration** of the dataset (~70 GB) available [`here`](https://bimcv.cipf.es/bimcv-projects/bimcv-covid19/#1590858128006-9e640421-6711).
2. Unzip the folder and move it to [`data/raw`](data/raw).
3. Untar all the `tar.gz` files in the folder.
4. Run the script available in [`src/adverx`](src/adverx/):

    ```bash
    python preprocess_dataset.py
    ```
5. You can delete the original dataset folder in [`data/raw`](data/raw).

## Documentation

Full documentation is available here: [`docs/`](docs).

## Dev

See the [Developer](docs/DEVELOPER.md) guidelines for more information.

## Contributing

Contributions of any kind are welcome. Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md]) for details and
the process for submitting pull requests to us.

## Changelog

See the [Changelog](CHANGELOG.md) for more information.

## Security

Thank you for improving the security of the project, please see the [Security Policy](docs/SECURITY.md)
for more information.

## License

This project is licensed under the terms of the `No license` license.
See [LICENSE](LICENSE) for more details.

## Citation

If you publish work that uses AdverX, please cite AdverX as follows:

```bibtex
@misc{AdverX,
  author = {TUe},
  title = {Official repository for AdverX.},
  year = {2024},
}
```
