# ABC Electronics Manufacturing - Operations Research Solver

A comprehensive Streamlit web application for solving Operations Research problems for ABC Electronics Manufacturing Company.

## 🎯 Features

### 1. **Simplex Method with Complete Sensitivity Analysis**
- **Big-M Method** implementation handling all constraint types:
  - `<=` (less-than-equal) constraints with slack variables
  - `>=` (greater-than-equal) constraints with surplus + artificial variables
  - `=` (equal) constraints with artificial variables
  
- **5 Types of Sensitivity Analysis** (activated when changes are made):
  1. **RHS Changes**: Impact of changing resource availability
  2. **Constraint Coefficient Changes**: Effect of changing consumption rates
  3. **Objective Function Coefficient Changes**: Reduced cost analysis
  4. **Adding New Constraint**: What-if scenario analysis
  5. **Adding New Variable**: New product profitability analysis

### 2. **Assignment Problem (Hungarian Method)**
- Default 10×10 matrix for worker-task assignment
- **Dynamic Matrix Sizing**:
  - Add complete rows (workers)
  - Add complete columns (tasks)
  - **Automatic square matrix balancing** (if 11×10, auto-pads to 11×11)
- Handles unbalanced problems with dummy workers/tasks

### 3. **Transportation Problem**
- **Least Cost Method** for initial basic feasible solution
- **MODI (UV) Method** for optimal solution
- **Automatic Degeneracy Detection and Handling**:
  - Detects when allocations < (m + n - 1)
  - Adds epsilon (ε) allocations to resolve degeneracy
- Balances unbalanced problems automatically

## 📊 Problem Data

### Linear Programming (Simplex)
**Products (10 Decision Variables):**
- X1: Power Banks ($25 profit)
- X2: USB Cables ($8 profit)
- X3: Bluetooth Speakers ($45 profit)
- X4: Headphones ($30 profit)
- X5: Chargers ($12 profit)
- X6: Adapters ($10 profit)
- X7: Smart Watches ($80 profit)
- X8: Keyboards ($35 profit)
- X9: Mouse ($15 profit)
- X10: Webcams ($50 profit)

**Constraints (10):**
1. Assembly Time ≤ 5000 hours
2. Raw Material ≤ 3500 kg
3. Labor Hours ≤ 4500 hours
4. Machine Hours ≤ 2800 hours
5. Quality Check Time ≤ 1500 hours
6. Packaging Materials ≤ 2500 units
7. Storage Space ≤ 8000 cubic ft
8. Electricity ≤ 6000 kWh
9. Budget ≤ $45,000
10. Minimum Total Production ≥ 500 units (Note: ≥ constraint)

### Assignment Problem
- 10 workers × 10 tasks
- Minimize total completion time
- Cost matrix represents time in hours

### Transportation Problem
- 10 factories (F1-F10) with total supply: 4500 units
- 10 stores (S1-S10) with total demand: 4500 units
- Balanced problem
- Cost matrix in $ per unit

## 🚀 Installation & Usage

### Prerequisites
```bash
Python 3.8+
pip install -r requirements.txt
```

### Required Packages
```txt
streamlit>=1.28.0
numpy>=1.24.0
pandas>=2.0.0
```

### Run Application
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 💡 How to Use

### Simplex Method
1. Navigate to "Simplex Method" in the sidebar
2. **To trigger sensitivity analysis**: Modify any values:
   - Edit objective function coefficients (profit per unit)
   - Change constraint types (<=, >=, =)
   - Modify RHS values (resource limits)
   - Edit constraint coefficients (consumption rates)
3. Click "🚀 Solve Simplex Problem"
4. View optimal solution and comprehensive sensitivity analysis

**Important Notes:**
- ≥ constraints are handled with surplus + artificial variables
- = constraints use artificial variables only
- Sensitivity analysis only activates when changes are detected

### Assignment Problem
1. Navigate to "Assignment Problem" in the sidebar
2. Default 10×10 matrix is loaded
3. **To add rows/columns**:
   - Click "➕ Add Complete Row" - auto-pads columns if needed
   - Click "➕ Add Complete Column" - auto-pads rows if needed
   - System maintains square matrix automatically
4. Edit cost matrix using the interactive data editor
5. Click "🚀 Solve Assignment Problem"

**Matrix Balancing Examples:**
- Start: 10×10
- Add row → 11×10 → **auto-pads to 11×11**
- Add column → 11×11 → 11×12 → **auto-pads to 12×12**

### Transportation Problem
1. Navigate to "Transportation Problem" in the sidebar
2. Edit supply, demand, or cost values if needed
3. Click "🚀 Solve Transportation Problem"
4. **Degeneracy handling** is automatic:
   - If allocations < (m+n-1), epsilon cells are added
   - Solution shows where ε allocations were placed
5. View initial solution (Least Cost) and optimal solution (MODI)

## 🔍 Key Features Explained

### Big-M Method (Simplex)
Handles all three constraint types properly:
```
<= : Add slack variable s_i
>= : Add surplus variable (−) + artificial variable A_i (Big-M penalty)
 = : Add artificial variable A_i (Big-M penalty)
```

### Sensitivity Analysis Activation
- **Automatic Detection**: Compares current values with original values
- **Triggers when**: Any coefficient, RHS, or constraint type changes
- **5 Comprehensive Types**: Covers all possible sensitivity scenarios

### Assignment Matrix Dynamics
```
Action: Add Row (11×10)
Result: Auto-add column → 11×11 (square maintained)

Action: Add Column (10×11)  
Result: Auto-add row → 11×11 (square maintained)
```

### Degeneracy in Transportation
```
Condition: Basic allocations < (m + n - 1)
Detection: Automatic count check
Resolution: Add ε = 0.0001 to strategic cells
```

## 📈 Sample Output

### Simplex Output
```
Maximum Profit Z = $147,500.00

Decision Variables (Product Units):
  X1_PowerBank: 125.50 units
  X2_USB: 0.00 units
  X3_Speaker: 450.25 units
  ...

Shadow Prices (Dual Values):
  Assembly Time: $12.5000
  Raw Material: $8.7500
  ...

SENSITIVITY ANALYSIS:
1️⃣ RHS SENSITIVITY
  Assembly Time (Current: 5000):
    Change by +100 → New Z ≈ $148,750.00
    ...
```

### Assignment Output
```
OPTIMAL ASSIGNMENT:
  Worker 1 → Task 3 | Time: 14.0 hours
  Worker 2 → Task 7 | Time: 15.0 hours
  ...
  
✓ Minimum Total Time: 165.0 hours
```

### Transportation Output
```
DEGENERACY DETECTED
✓ Added ε to cell (F3, S5)

OPTIMAL ALLOCATION:
  Factory_1 → Store_1: 450 units @ $8.0 = $3600.00
  ...
  
✓ Minimum Total Transportation Cost: $41,250.00
```

## 🛠️ Technical Implementation

### Algorithms Used
1. **Simplex Method**: Two-Phase/Big-M with pivot operations
2. **Hungarian Algorithm**: Row/column reduction with optimal assignment
3. **Least Cost Method**: Greedy initial allocation
4. **MODI Method**: Dual variables (u, v) with opportunity cost calculation

### Data Structures
- NumPy arrays for matrix operations
- Pandas DataFrames for display
- Streamlit session state for data persistence

## 📝 Notes

### Constraint Handling
- The program correctly differentiates between `<=`, `>=`, and `=`
- ≥ constraints require **both** surplus and artificial variables
- Big-M value is set to 1,000,000 (adjustable if needed)

### Matrix Operations
- All matrix operations use NumPy for efficiency
- Automatic type conversion to float64 for numerical stability
- Epsilon value (1e-6) used for zero comparisons

### UI/UX
- Collapsible expanders for data editing
- Real-time change detection for sensitivity analysis
- Color-coded status indicators
- Interactive data editors with pandas integration

## 🐛 Troubleshooting

**Issue**: Sensitivity analysis not showing
**Solution**: Make sure to modify at least one value (objective, RHS, coefficient, or constraint type)

**Issue**: Assignment matrix not square
**Solution**: Use "Add Row" or "Add Column" buttons - auto-balancing will handle it

**Issue**: Transportation solution infeasible
**Solution**: Check supply and demand totals - unbalanced problems are auto-fixed with dummy source/destination

## 👨‍💻 Development

**Version**: 2.0  
**Language**: Python 3.8+  
**Framework**: Streamlit  
**Date**: December 2025

**Files**:
- `app.py` - Main application (complete version)
- `app_backup.py` - Original backup
- `requirements.txt` - Dependencies
- `README.md` - Documentation

## 📄 License

Educational project for ABC Electronics Manufacturing Company - Operations Research course.

## 🎓 References

- Simplex Method with Big-M
- Hungarian Algorithm for Assignment
- MODI Method for Transportation
- Sensitivity Analysis in Linear Programming
- Degeneracy in Transportation Problems

---

**Happy Optimizing! 🚀📊**
