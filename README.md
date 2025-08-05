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

Here we discuss different parts of the code. From the OpenFOAM side, the folders `0`, `constant`, and `system` should be present, as they contain the initial conditions, physical properties, and simulation setup, respectively. Additionally, the code file `of_pybind11_system.C` builds the bridge between the PINN code written in PyTorch and OpenFOAM.

The overall folder structure looks like this:

```text
📁 NCL_Equation_Example
├── 📁 0
│   └── <initial condition files>
├── 📁 constant
│   └── <physical property files>
├── 📁 system
│   └── <simulation control files>
├── 📁 Make
│   └── <pybind build configuration files>
├── 🧠 of_pybind11_system.C
├── 🐍 run_NCL_DisPINN.py (Main Script)

```






