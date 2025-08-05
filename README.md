## 🔗 PINNFOAM - Coupling Full-Order and Reduced-Order Model with PINN

We aim to couple the Full Order Model (FOM) and the Reduced Order Model (ROM), both of which are computed using the Finite Volume-based solver **OpenFOAM**.  For background and preliminary work, we encourage readers to consult the authors' initial study:  
👉 [https://doi.org/10.1016/j.camwa.2025.05.004](https://doi.org/10.1016/j.camwa.2025.05.004)
And the available Codes: 
👉 https://github.com/rahulhalderAERO/Paper-DisPINN1

## FOM + PINN

In the current codes, we first demonstrate how to couple a full-order model (FOM) with a Physics-Informed Neural Network (PINN) using an example of a nonlinear conservation law equation.

### Step 1

Go inside the folder `NCL-FOM` and install PINA by running:

```bash
pip install -e.
```
### Step 2

Go to the folder `NCL_Equation_Example` and run the script `run_burgers_Dis_ANN.py` with:

    python3 run_burgers_Dis_ANN.py -s 0 0

## Code Structure

Here we discuss different parts of the code. From the OpenFOAM side, the folders `0`, `constant`, and `system` should be present, as they contain the initial conditions, physical properties, and simulation setup, respectively. Additionally, the code file `of_pybind11_system.C` builds the bridge between the PINN code written in PyTorch and OpenFOAM. The `problems/` folder contains the Python-side problem definitions, which describe how the PINN is coupled with the nonlinear conservation law (NCL) as solved in OpenFOAM. This folder is essential for defining the PINN structure, loss formulation, and data interfacing.

The overall folder structure looks like this:

```text
📁 NCL-FOM
├── 📁 0
│   └── <initial condition files>
├── 📁 constant
│   └── <physical property files>
├── 📁 system
│   └── <simulation control files>
├── 📁 Make
│   └── <OpenFOAM build configuration files>
├── 📁 problems
│   └── <Python definitions for PINN–OpenFOAM coupling>
├── 🧠 of_pybind11_system.C
├── 🐍 run_NCL_DisPINN.py
└── 📄 other_code_files...
```
## C++–Python Coupling (of_pybind11_system.C)

Next, we briefly look into different parts of the `of_pybind11_system.C` code. The `A` matrix, which appears in this file, is obtained from the linearised discretised form of the nonlinear conservation law (NCL), as shown in Equation (15) of the paper. This matrix is central to forming the system of equations passed between OpenFOAM and the PyTorch-based PINN. The code uses `pybind11` to export the matrix `A` and `b` from OpenFOAM to the PINA code.

```cpp
Eigen::SparseMatrix<double> get_system_matrix(Eigen::VectorXd& U)
{
    Foam2Eigen::Eigen2field(_U(), U);
    fvVectorMatrix UEqn(
        fvm::ddt(_U()) +
        fvc::div(_phi(), _U()) -
        fvm::laplacian(_nu(), _U())
    );

    Eigen::SparseMatrix<double> A;
    Eigen::VectorXd b;
    Foam2Eigen::fvMatrix2Eigen(UEqn, A, b);
    return A;
}

Eigen::VectorXd get_rhs(Eigen::VectorXd& U)//,Eigen::VectorXd& S)
    {
        Foam2Eigen::Eigen2field(_U(), U);
        fvVectorMatrix UEqn(fvm::ddt(_U()) + fvc::div(_phi(), _U()) - fvm::laplacian(_nu(),_U()));
        Eigen::SparseMatrix<double> A;
        Eigen::VectorXd b;
        Foam2Eigen::fvMatrix2Eigen(UEqn, A, b);
        return b;
    }

```

## Python–OpenFOAM Integration (NCL_tensor_discrete_ANN.py)

The part of the code responsible for calling the `A` matrix and `b` vector from the Python side is located in:
```text
📁 problems
└── 📄 NCL_tensor_discrete_ANN.py
```


This script serves as the Python interface for interacting with the linearised system assembled in OpenFOAM. It calls the C++ function `get_system_matrix(...)` (exposed via `pybind11`) to fetch the current state of the system matrix `A` and the corresponding right-hand side vector `b`.These matrices and vectors are then used in the PINN's loss formulation to enforce consistency with the discretised PDE. This tight coupling allows the neural network to learn from and adapt to the dynamics governed by the OpenFOAM solver in real time. This integration is critical for implementing discrete PINNs (DisPINNs), where the loss is built directly from the residuals of the numerical system, ensuring physics fidelity without symbolic differentiation.

```python
A = a.get_system_matrix(U)
A_array = A.toarray()
b = a.get_rhs(U)
AR_tensor = torch.from_numpy(A_array).float()
bR_tensor = torch.from_numpy(b).float().reshape(-1, 1)
```


## ROM + PINN

In the current codes, we first demonstrate how to couple a reduced-order model (FOM) with a Physics-Informed Neural Network (PINN) using an example of a Flow past an incompressible flow. 

