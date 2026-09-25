# HBJ_train.py

HJB value-network training script.

Objective:
    Train a HJB-style neural value function that estimates the remaining collision time/TTC over the OVM state domain through a strong fixed-point approach

Required before running:
    None, except the Python packages listed in requirements.txt.

Main outputs:
    outputs_hjb_k1_nosup/value_net.keras
    outputs_hjb_k1_nosup/value_net_latest.keras
    outputs_hjb_k1_nosup/history.json
    outputs_hjb_k1_nosup/run_config.json
    outputs_hjb_k1_nosup/boundary_table_line.npz


Run order:
    Run this script first. Later scripts load outputs_hjb_k1_nosup/value_net.keras.
