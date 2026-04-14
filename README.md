# pySWApp - an interactive, open-source python toolbox for processing seismic surface wave data

by
Nathalie Roser, Ilaria Barone, Alberto Carrera and Adrián Flores Orozco

---

## Abstract

We introduce the open-source python library pySWApp, which provides a flexible 
and semi-interactive framework for managing and processing 2D active seismic 
surface wave data for dispersion curve analysis. Among classical approaches for 
surface-wave analysis such as the Multichannel Analysis of Surface Waves (MASW), 
pySWApp encompasses advanced approaches for extracting dispersion curves under 
laterally challenging conditions: the Multi-Offset Phase Analysis (MOPA) and the 
Tomographic-Like Approach (Tomo2D).

---

## Folder and File Structure

The source code of the pySWApp library is in the `code` folder.
Synthetic data to reproduce the exemplary use cases presented in the 
manuscript are provided in the `data` folder. 
Exemplary scripts to showcase the libraries key functions are  provided in 
the `examples` folder. 
The pdf of the manuscript are in the `docs` folder.

---

### **Library: `pyswapp`**
Contains the core processing routines:
- **`manager.py`**  
  Classes tailored for processing active 1D and 2D surface-wave analysis.
- **`stream.py`**  
  Class for handling seismic shot gathers.
- **`curve.py`**  
  Class for handling dispersion curves.
- **`curves.py`**  
  Class for combining multiple dispersion curves.
- **`qtapps.py`**  
  PyQT5 apps for interactive data handling.
- **`utils`**  
  A set of utility functions for import and export of files, plotting, physical relations, 
  and solving linear mathematical systems.

---

### **Examples**
Scripts and Jupyter Notebooks demonstrating various use cases:
- **`0_create_geometry`**  
  Create and read a `geometry.csv` file.
- **`1_manager_basics`**  
  An overview of basic capabilities applied to a single shot gather.
- **`2a_run_MASW2D`**  
  Exemplary use of the MASW2D Manager.
- **`2b_run_MASW2D_windowing`**  
  Exemplary use of the MASW2D Manager with additional spatial windowing.
- **`3_run_tomo2d`**  
  Exemplary use of the Tomo2D Manager.
- **`4a_combine_curves`**  
  Dispersion curve combination.
- **`4b_process_curves`**  
  Post-processing of dispersion curves.

---

### **Data: `data`**
Directory for raw and processed data:
- **`syn_data/`**  
  Synthetic data set.
  - **`proc/`**  
    Processed data.
    - **`1_basics/`**  
      Project folder containing processing outputs.
  - **`raw/`**  
    Original raw data.

---

## Dependencies

You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.

The required dependencies are specified in the file `environment.yml`.

Open a terminal (Linux & Mac) or the Anaconda Prompt (Windows) and run the 
following command in the repository folder (where `environment.yml`
is located) to create a new environment and install the required
dependencies:

    conda env create

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/naroser/pyswapp.git

2. Installing the library

    To use the pySWApp library we suggest to install it in an conda 
    environment. Run the following lines to activate the corresponding environment 
    and start the setup process.
    ```bash
    conda activate <env_name>
    cd code
    pip install .

## Reference

    Roser, N., Barone, I., Carrera, A., Flores Orozco, A., 2026. 
    pySWApp - An interactive, open-source Python toolbox for processing seismic surface wave data.  
    Computers and Geosciences
    [In Preparation]

## License

All source code is made available under the MIT License. See LICENSE.md for 
the full license text.
