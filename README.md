# Power Flow Solver

A Python implementation of the **Newton-Raphson** method for solving power flow analysis on a multi-bus system. Developed as a final project for **ECEN4293 - Python with Numerical Methods** at Oklahoma State University.

---

## 📋 Project Overview

This project implements a robust **Newton-Raphson power flow solver** using **JAX** for automatic Jacobian computation via automatic differentiation. The solver analyzes steady-state operating conditions of the IEEE 9-bus transmission system, calculating:

- Bus voltage magnitudes (Vm) and phase angles (Va)
- Real and reactive power injections
- Slack bus power contribution
- Transmission line power flows and system losses

### Key Features

- **Modern implementation** using JAX + NumPy for fast and accurate Jacobian calculation
- **Automatic differentiation** (no manual Jacobian coding required)
- Support for Slack (PV), PV, and PQ buses
- Modular design with clear separation of concerns
- JSON-based input for easy data modification
- Damping factor support for improved convergence
- Detailed output including branch flows and total system losses

---

## 🛠️ Technologies Used

- **Python 3**
- **JAX** (with 64-bit precision enabled)
- **NumPy**
- **JSON** for data input
- **argparse** for command-line interface
