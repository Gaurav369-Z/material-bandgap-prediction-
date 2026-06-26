# material-bandgap-prediction

# Material Band Gap Prediction Using Machine Learning: Comparison with Quantum ESPRESSO and Experimental Data

##Overview

This project predicts electronic band gap of crystalline material using machine learning model, to trained model JARVIS DFT dataset is used. This project uses Random Forest regression model with engineered material descriptors such as density, formation energy, elastic properties, and atomic features, further machine learning predictions are compared with Density Functional Theory (DFT) calculations performed using Quantum ESPRESSO and also with experimental band-gap values.
The objective is to investigate how accurately a machine learning model can reproduce DFT-calculated band gaps while significantly reducing computational cost.

##Dataset

This project uses the JARVIS DFT 3D Dataset.

##Target Property

optb88vdw_bandgap

## Input Features

- Density
- Formation Energy per Atom
- Energy Above Hull
- Number of Atoms (nat)
- Space Group Number
- Bulk Modulus
- Shear Modulus
- Average Atomic Number
- Average Electronegativity
- Electronegativity Difference
- Average Atomic Radius
-Average Atomic Radius

## Machine Learning Model

Random Forest Regressor

## Model Performance
## Model Performance

| Metric | Value |
|--------|------:|
| R² Score | 0.7915 |
| Mean Absolute Error (MAE) | 0.3256 eV |

## Future Work

- Compare Random Forest predictions with Quantum ESPRESSO calculations.
- Validate predictions using experimental band-gap values.
