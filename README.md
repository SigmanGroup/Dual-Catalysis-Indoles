# Dual-Catalysis-Indoles

This repository contains the code for calculating the oxidation potential of indole compounds and to extract their stereoelectronic properties for chemical space visualization and modeling. For in depth information, please refer to the manuscript and corresponding Supporting Information available at: (insert link)

The files, directories, and their respective contents are listed below:

1. ORCA_oxidation_potential_calculator.py: python script to calculate the oxidation potential of a series of indole compounds from the corresponding DFT calculations in ORCA.

2. get_properties_ORCA.ipynb: Jupyter notebook for extracting the stereoelectronic properties of the calculated indoles from ORCA output files. This notebook has been adapted from the original Gaussian version of the Sigman's group "GetProperties". For the
original script, please visit the [original repository](https://github.com/SigmanGroup/GetProperties.git).

3. GetPropertiesORCA: folder containing the python scripts needed to run the Jupyter notebook.

4. example_files: folder containing some examples of the ORCA output files (OPT and SPC) and coordinates (.xyz) of calculated indoles to test the oxidation potential calculator scrit and the Jupyter notebook.

5. results: folder containing the CSV / Excel files obtained from the Jupyter notebook.

6. README.md: instructions file.

## Installation

The required environment for this project, named 'gpenv_312', is obtained from the Sigman's group "GetProperties" repository and can be downloaded from [original repository](https://github.com/SigmanGroup/GetProperties.git).

1. Download the .yml file and create a conda environment:

    `conda env create -f gpenv_312.yml`

2. Activate the environment.

    `conda activate gpenv_312`

3. Clone the repository.

    `git clone https://github.com/cristianrosso92/Dual-Catalysis-Indoles.git`

4. Navigate into the project directory.

    `cd Dual-Catalysis-Indoles`

## Usage of the oxidation potential calculator script

To use the Python script `ORCA_oxidation_potential_calculator.py` to calculate the oxidation potentials of the indole compounds, please follow this checklist:

1. Make sure that all the ORCA `.out` files for both geometry optimizations (OPT) and single-point calculations (SPC) are present in the same folder as the Python script.

2. The files must be named as follows:

    - `S(number)_OPT.out`: geometry optimization of the neutral indoles
    - `O(number)_OPT.out`: geometry optimization of the oxidized indoles
    - `S(number)_SPC.out`: single-point calculation of the neutral indoles
    - `O(number)_SPC.out`: single-point calculation of the oxidized indoles

3. Run the script:

    `python3 ORCA_oxidation_potential_calculator.py`

4. The script creates an Excel file named `oxidation_potential_results.xlsx`, containing the calculated oxidation potentials in V vs SHE and V vs SCE. The file also includes a column reporting the status of the DFT calculations, e.g. `OK`, together with the individual values extracted from the different `.out` files, which can be used to double-check the integrity of the results.

## Usage of the Jupyter notebook to extract properties

Use the notebook `get_properties_ORCA.ipynb` to collect the properties of the neutral indoles `S(number)`. Be sure to set your Jupyter kernel to `gpenv_312`. Please follow this checklist:

1. Make sure that all the ORCA `S*_SPC.out`, `S*_SPC.xyz`, and `S*_OPT.out` files are present in the same folder as the Jupyter notebook. It is recommended to run the notebook in a clean folder containing only the ORCA files that should be analyzed, and not directly in the same folder used for the oxidation potential calculation. The notebook does not filter files by prefix; therefore, any undesired compatible file might be processed, possibly leading to property-extraction errors.

2. Follow the instructions provided in each cell of the notebook. The `.xyz` files will be converted into the corresponding `.mol` files, which will then be used for RDKit substructure matching and atom mapping.

3. Dedicated cells are provided at the end of the notebook to handle conformers and symmetric molecules.

## Contributors

- Cristian Rosso, PhD
- Raymond Z. Lo
