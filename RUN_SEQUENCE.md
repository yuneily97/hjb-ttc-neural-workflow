# Run Sequence

Run these commands from the cleaned project folder.

```bash
python HBJ_train.py
python HBJ_Validation.py
python NN_sol_training_dataset.py
python NN_V.py
python NN_X.py
python FG_sim_vs_NN_sim.py
python FG_sim_vs_NN_based_approx.py
python NN_sol_nonlinear_stability.py
```

`HBJ_Validation.py`, `FG_sim_vs_NN_sim.py`, `FG_sim_vs_NN_based_approx.py`, and `NN_sol_nonlinear_stability.py` are evaluation/diagnostic scripts. The minimum training pipeline is:

```bash
python HBJ_train.py
python NN_sol_training_dataset.py
python NN_V.py
python NN_X.py
```
