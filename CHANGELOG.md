# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/)
and semantic versioning principles.

---

## v0.3.0-beta — 2025-12-16

### Added
- Local **Gradio-based application** for MRgFUS temperature estimation.
- **Automatic temperature prediction** from power, duration, and skull parameters.
- **Regime-dependent uncertainty band (± MAE)** displayed with each prediction.
- **Trajectory visualization** for multiple sonication durations.
- Interactive **X-axis toggle**: Energy (kJ) ↔ Power (W).
- **Smooth logarithmic fit** option for trajectory visualization.
- **Target planner (inverse planning)** reporting required power for:
  - 50 °C
  - 55 °C
  - 60 °C
- **Sonication logging** including:
  - Timestamp
  - Sonication ID
  - Power, duration, energy
  - Predicted temperature
  - Real peak average temperature
  - Prediction error (Real − Predicted)
- **Save & Exit** functionality:
  - Exports CSV and PNG summaries to `treatmentlogs/`
  - Safely terminates the local Gradio server
  - Prevents orphan Python processes

### Changed
- Switched to a **minimal feature set model** for robustness and portability.
- Added **temperature-regime–aware MAE** instead of a single global error.
- Improved UI layout and automatic refresh behavior.
- Standardized model metadata and environment reproducibility.

### Fixed
- Model incompatibility issues due to scikit-learn version mismatch.
- Static trajectory plots not updating with parameter changes.
- Feature name warnings during inference.
- Server process remaining active after browser close.

### Known limitations
- Research / engineering tool only.
- Not validated for clinical decision making.
- Model performance degrades for extreme temperature regimes (≥60 °C).

---

## v0.2.0-beta — 2025-11-30

### Added
- Initial temperature prediction model.
- Basic local UI for parameter input.
- Manual prediction workflow.

---

## v0.1.0-alpha — 2025-10-XX

### Added
- Prototype data processing pipeline.
- Exploratory model training and validation.
