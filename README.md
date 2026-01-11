# AeroML: Inverse Airfoil Design using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 🎯 Overview

AeroML replaces computationally expensive CFD simulations with a fast ML-based inverse design system. Input desired aerodynamic performance metrics (drag coefficient, lift coefficient, pressure distribution) and generate corresponding airfoil geometries in seconds instead of hours.

**Key Innovation:** Combines an autoencoder for geometry compression with an inverse neural network that maps performance → latent codes → airfoil shapes.

## 🏗️ Architecture

```
Input: Aerodynamic Parameters (Cd, CL, CM, pressure × 1000 points)
   ↓
[Inverse Neural Network: 1007 → 1024 → 1024 → 512 → 6]
   ↓
Latent Space (6D representation)
   ↓
[Autoencoder Decoder: 6 → 10 → 18]
   ↓
Output: CST Coefficients → Airfoil Geometry
```

**Models:**
- **Autoencoder:** Compresses 18 CST coefficients → 6D latent space 
- **Inverse Network:** Maps 1007 aerodynamic features → 6D latent codes 

## 📊 Results

| Metric | Value |
|--------|-------|
| Reconstruction Accuracy | 90% |
| Validation Loss (Autoencoder) | 0.0179 |
| Validation Loss (Inverse Network) | 0.3106 |
| Dataset Size | 200 CFD simulations |

### Prerequisites
```bash
Python 3.8+
PyTorch 2.0+
NumPy, Pandas, Matplotlib
AirfRANS dataset
```

## 🔬 Methodology

1. **Data Preprocessing:** Convert heterogeneous CFD mesh data (varying point counts) to standardized 18 CST coefficients using Class Shape Transformation
2. **Autoencoder Training:** Learn 6D latent representation of airfoil geometries (500 epochs, MSE loss)
3. **Inverse Network Training:** Map aerodynamic parameters to latent space (300 epochs, L2 regularization)
4. **Validation:** 80/20 train-test split, PCA manifold analysis to ensure generated geometries are realistic

## 📚 Dataset

This project uses the [AirfRANS dataset](https://github.com/Extrality/AirfRANS_dataset) by Bonnet et al. (2022):
- 1,000 NACA airfoil simulations
- Reynolds numbers: 2-6 × 10⁶
- Angles of attack: 5-15°
- Training subset: 200 simulations ('scarce' regime)

## 👥 Contributors

**Group 4 - MAE 551 Fall 2025:**  
Dan Hlevca, Ryan Yeager, Trevor Dyhrkopp, Vedant Bhat, Vishavjit Singh Khinda

## 🔗 References

Bonnet, F., Mazari, J., Cinnella, P., & Gallinari, P. (2022). *Airfrans: High fidelity computational fluid dynamics dataset for approximating reynolds-averaged navier–stokes solutions.* Advances in Neural Information Processing Systems, 35, 23463-23478.

