# Cleaned Traffic Safety / HJB-TTC Neural Workflow

This folder contains cleaned versions of the original Python scripts. The executable code was preserved as much as possible: existing comments and commented-out blocks were removed, and a concise module-level description was added to the top of each file.

## Setup

Create and activate a Python environment, then install the required packages:

```bash
pip install -r requirements.txt
```

The scripts were written for TensorFlow/Keras workflows and save/load `.keras` and `.h5` model files. Several scripts use `scipy.integrate.cumtrapz`, so `requirements.txt` pins SciPy below the versions where that alias may be unavailable.

## Recommended run sequence

1. `HBJ_train.py`
   - Trains the HJB/value-network TTC estimator.
   - Produces `outputs_hjb_k1_nosup/value_net.keras`.

2. `HBJ_Validation.py` optional but recommended
   - Validates the trained HJB/value network against ODE-based ground truth.
   - Requires `outputs_hjb_k1_nosup/value_net.keras`.

3. `NN_sol_training_dataset.py`
   - Generates `proposed_data_long_full.npz` for the neural trajectory models.
   - Requires `outputs_hjb_k1_nosup/value_net.keras`.

4. `NN_V.py`
   - Trains the first-stage neural trajectory model for `[X(t), V(t)]`.
   - Requires `outputs_hjb_k1_nosup/value_net.keras` and `proposed_data_long_full.npz`.
   - Produces `trained_proposed_model_semi_supervised3_full_gt_sine_uniform_concent_on_close_just_leaky_relu_in_front_omega15_4_full.h5`.

5. `NN_X.py`
   - Trains the second-stage spacing-only model `X(t)`.
   - Requires the HJB/value model, the dataset, and the first-stage model from `NN_V.py`.
   - Produces `trained_proposed_model_semi_supervised3_full_gt_sine_uniform_concent_on_close_just_leaky_relu_in_front_part2_omega15_4_full.h5`.

6. `FG_sim_vs_NN_sim.py` optional evaluation
   - Compares full ground-truth simulation against simulation using the HJB/value-network TTC estimator.
   - Requires `outputs_hjb_k1_nosup/value_net.keras`.

7. `FG_sim_vs_NN_based_approx.py` optional evaluation
   - Compares full ground-truth simulation with the two-step neural approximation.
   - Requires the HJB/value model and both trained trajectory models.

8. `NN_sol_nonlinear_stability.py` optional downstream simulation
   - Runs nonlinear stability simulations using the HJB/value model and both trained neural models.
   - Requires the HJB/value model and both trained trajectory models.

## File guide

| File | Purpose | Required inputs | Main outputs |
|---|---|---|---|
| `HBJ_train.py` | Train HJB/value TTC estimator | Python dependencies | `outputs_hjb_k1_nosup/value_net.keras` |
| `HBJ_Validation.py` | Validate HJB/value TTC estimator | `outputs_hjb_k1_nosup/value_net.keras` | validation CSV and plots |
| `NN_sol_training_dataset.py` | Generate neural trajectory training dataset | `outputs_hjb_k1_nosup/value_net.keras` | `proposed_data_long_full.npz` |
| `NN_V.py` | Train first-stage `[X,V]` model | HJB model, `proposed_data_long_full.npz` | first-stage `.h5` model, plots, CSV |
| `NN_X.py` | Train second-stage `X` model | HJB model, dataset, first-stage `.h5` model | second-stage `.h5` model, plots, CSV |
| `FG_sim_vs_NN_sim.py` | Compare FG simulation and HJB-TTC simulation | HJB model | phase portrait plot |
| `FG_sim_vs_NN_based_approx.py` | Compare FG simulation and two-step NN approximation | HJB model, first-stage `.h5`, second-stage `.h5` | phase portrait/TTC plots |
| `NN_sol_nonlinear_stability.py` | Multi-vehicle nonlinear stability simulation | HJB model, first-stage `.h5`, second-stage `.h5` | stability profiles and plots |

## Important path expectations

The scripts expect to be run from the same parent directory where their generated files are located. In particular, several scripts load:

```text
outputs_hjb_k1_nosup/value_net.keras
proposed_data_long_full.npz
trained_proposed_model_semi_supervised3_full_gt_sine_uniform_concent_on_close_just_leaky_relu_in_front_omega15_4_full.h5
trained_proposed_model_semi_supervised3_full_gt_sine_uniform_concent_on_close_just_leaky_relu_in_front_part2_omega15_4_full.h5
```

If a model-loading error occurs, first check that the preceding script in the run sequence completed successfully and that the current working directory is correct.

## Notes

- The cleaned scripts do not include the original commented-out alternative architectures or old paths.
- The core executable statements were not intentionally changed.
- Long simulations/training jobs may take significant time depending on the number of samples, time grid resolution, and CPU/GPU availability.
