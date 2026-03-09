# TPU Wind Pressure Sequence Inpainting

This repository studies conditional sequence inpainting on TPU wind-pressure tap data.
The goal is to reconstruct the back half of a 3-second pressure window from:

- the observed front half of the same window
- surface metadata (`face_id`)
- tap location metadata (`left_dist`, `bottom_dist`)

In practical terms, this project asks:

> Can a conditional generative model recover the missing rear segment of a wind-pressure time series better than simple supervised baselines?

## Research Summary

The project compares three model families on the same task:

- Diffusion-based conditional generator
- Flow-matching-based conditional generator
- Simple baselines (`ANN`, `CNN`)

The main evaluation setting is:

1. Split each tap signal into fixed 3-second windows.
2. Normalize using train-split statistics only.
3. Keep only the front half as observed context at test time.
4. Predict or generate the back half.
5. Compare reconstruction quality with back-half MSE and a proxy FID-style metric.

This makes the repository closer to an applied research prototype than a generic training template.

## Dataset and Labels

Raw data is expected under `tpu_data/` as MATLAB `.mat` files.
From each file, the preprocessing step reads:

- `Wind_pressure_coefficients`
- `Location_of_measured_points`
- `Sample_frequency`

The location metadata is interpreted as:

- row `0`: left distance
- row `1`: bottom distance
- row `2`: point id
- row `3`: face id

Face ids correspond to building surfaces:

- `1`: windward
- `2`: right sideward
- `3`: leeward
- `4`: left sideward

## Current Pipeline

The current codebase already implements an end-to-end experimental flow:

`raw .mat -> preprocess -> save processed dataset -> train model -> evaluate reconstructions`

Key modules:

- `params.py`: central runtime and experiment configuration
- `data_preprocess.py`: MATLAB loading, windowing, stratified train/test split, normalization
- `data_save.py`: processed dataset persistence and validation
- `model_DM.py`: conditional diffusion model
- `model_FM.py`: conditional flow matching model
- `baseline_models.py`: ANN/CNN back-half predictors
- `train_DM.py`: diffusion training entrypoint
- `train_FM.py`: flow matching training entrypoint
- `train_baselines.py`: ANN/CNN training entrypoint
- `test.py`: evaluation and comparison script

## Repository Status

This repository is already beyond scaffolding stage.
The local workspace contains:

- processed dataset artifacts
- trained checkpoints for diffusion / flow / ANN / CNN
- saved training metrics
- saved test metrics and reconstruction outputs

That means the code has already been used for at least one experimental run.

## How to Run

This project is currently parser-free and configuration-driven through `params.py`.
The main executable scripts are:

```bash
python train_DM.py
python train_FM.py
python train_baselines.py
python test.py
```

## Research Objective

From a research perspective, this repository investigates:

- conditional generation for partially observed wind-pressure sequences
- whether diffusion and flow matching outperform simple deterministic regressors
- whether reconstruction quality improves when front-half statistical constraints are enforced during sampling

## Notes

- `spec/spec.json` records the experiment intent, but the actual runtime source of truth is `params.py`.
- Large raw data, artifacts, and reference PDFs should usually not be committed to GitHub.
- A future cleanup step would be to standardize the training entrypoint as a single `train.py`.
