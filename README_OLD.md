# Operations Research Solver - Streamlit Web Application

## ABC Electronics Manufacturing Company

A comprehensive web-based Operations Research solver for:
- **Simplex Method** - Linear Programming with complete sensitivity analysis
- **Assignment Problem** - Hungarian Method
- **Transportation Problem** - Vogel's Approximation Method

## Features

### Simplex Method
- Customizable objective function (minimum 10 variables)
- Dynamic constraints with <=, >=, = operators
- Complete sensitivity analysis including:
  - RHS changes
  - Objective function coefficient changes
  - Constraint coefficient changes
  - Adding new constraints
  - Adding new variables
- Real-time input validation

### Assignment Problem
- Minimum 10x10 cost matrix
- Hungarian Method implementation
- Handles unbalanced problems automatically
- Default matrix or custom input

### Transportation Problem
- Multiple sources and destinations
- Vogel's Approximation Method
- Automatic balancing
- Default values or custom input

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Deployment

### Deploy to Streamlit Cloud
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository
4. Select `app.py` as the main file
5. Deploy!

### Deploy to Other Platforms
- **Heroku**: Add `setup.sh` and `Procfile`
- **Railway**: Connects directly to GitHub
- **Render**: Auto-detects Streamlit apps

## Usage

1. Select problem type from sidebar
2. Input parameters (all editable in UI)
3. Click solve button
4. View detailed solution with step-by-step process

## Requirements
- Python 3.8+
- Streamlit
- NumPy
- Pandas

## License
MIT License - Feel free to use and modify
