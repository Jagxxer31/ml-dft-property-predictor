# ML-Accelerated DFT: Faster Prediction of Molecular Energy Gaps

This project uses machine learning to make advanced quantum chemistry calculations much faster.

---

## The Problem

Accurate methods (like HSE) give good predictions for molecular energy gaps, but they are very slow and computationally expensive.  
Faster methods (like PBE) are cheaper but less accurate.

---

## The Solution

It trains a machine learning model to learn the difference between the fast method and the accurate method.

So instead of running the expensive calculation every time, we:

- Run the fast calculation (PBE)  
- Use machine learning to correct it  
- Get a result close to the expensive HSE calculation  

This gives near-high accuracy at much lower cost.

---

## Tech Used

- Gaussian (quantum chemistry calculations)  
- Python  
- NumPy & Pandas  
- scikit-learn  
- XGBoost  
- Neural Networks  
- Matplotlib for visualization  

---

## Workflow

Quantum calculations → Extract features → Train ML model → Predict corrected energy gaps → Compare with reference values

---

## Why It Matters

This approach can:

- Save large amounts of computational time  
- Enable faster molecular screening  
- Make advanced simulations more practical  
