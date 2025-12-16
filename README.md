# MRgFUS TempAI  
**v0.3.0-beta — Research Use Only**

Predictive and inverse-planning tool for **MR-guided Focused Ultrasound (MRgFUS)** temperature estimation, designed for research and clinical decision support.

---

## Overview

**MRgFUS TempAI** provides:
- Real-time temperature prediction using trained ML models
- Trajectory visualization across power / duration / energy
- Inverse planning (target temperature → required power or duration)
- Structured sonication logging with automatic export (CSV + PNG)

The tool is intended for **research and educational purposes** and **clinical decision support only**.

---

## Clinical Disclaimer

This software:
- Is **NOT** a certified medical device
- Has **NOT** been validated by any regulatory authority (FDA, EMA, CE)
- Must **NOT** be used as the sole basis for diagnosis or treatment

Any clinical use must be:
- Experimental
- Supervised by qualified medical professionals
- Used strictly as **secondary decision support**

See the `LICENSE` file for full legal terms.

---

## Key Features

- Non-linear and linear ML temperature models
- Automatic MAE reporting by temperature range
- Power ↔ Duration inverse planning with safety bounds
- Log-fit or raw model trajectory visualization
- Persistent sonication logging
- Clean Gradio-based local UI

---

## Models

Two trained models are included:
- **Non-Linear Model** (Gradient Boosting Regressor)
- **Linear Regression Model**

Both operate on the same minimal feature set:
- Power
- Duration
- Energy
- Skull area
- Mean SDR
- Number of active elements

---

## Local Installation

### 1 Clone repository
```bash
git clone https://github.com/<your-username>/MRgFUS-TempAI.git
cd MRgFUS-TempAI
```

### 2 Create Conda environment
```bash
conda env create -f environment.yml
conda activate tempai
```

### 3 Create Conda environment
Run the app
```bash
python tempai_advmodel_local.py
```
The interface opens automatically at:
http://127.0.0.1:7860

## Output Files

On exit, the application saves:
treatmentlogs/sonication_log_YYYYMMDD_HHMMSS.csv
treatmentlogs/sonication_summary_YYYYMMDD_HHMMSS.png

## Output Files

On exit, the application saves:

- treatmentlogs/sonication_log_YYYYMMDD_HHMMSS.csv  
- treatmentlogs/sonication_summary_YYYYMMDD_HHMMSS.png

## Inverse Planning Logic (Summary)

- Fix Duration: compute required Power to reach target temperature  
- Fix Power: compute required Duration to reach target temperature  

Numerical bisection is used directly on the ML model output (no linearization).

Details are documented in the Wiki.

## Project Status

Stage: Research / Pre-clinical  
Stability: Beta  
Validation: Not clinically validated  
Roadmap: Model refinement, uncertainty estimation, prospective validation  

## Author

Jose Angel Pineda Pardo  
Researcher — Neuroimaging & Therapeutic Ultrasound  

## License

Research & Clinical Decision Support License  
See LICENSE for full terms.

---

## Wiki

In GitHub → Wiki, create the following pages:

### Home
- Project overview
- Intended use
- High-level architecture

### Modeling
- Features
- Model choice
- Error behavior by temperature range

### Inverse Planning
- Why bisection
- Why not linear interpolation
- Safety bounds
- NA conditions

### Visualization
- Trajectories
- Log-fit vs raw model
- Interpretation guidelines

### Clinical Safety
- Decision support philosophy
- Known limitations
- Non-validation statement

---

## CI (Continuous Integration)

CI verifies that the repository is in a consistent and runnable state.

It performs:
- Basic import checks
- Syntax validation
- Automatic execution on push and pull requests

### Create CI structure

```bash
mkdir -p .github/workflows
touch .github/workflows/ci.yml
```