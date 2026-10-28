#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
python exp1_cycle_expressivity.py        | tee log_E1.txt
python exp2_perturbation_common_basis.py | tee log_E2.txt
python exp3_multihead.py                 | tee log_E3.txt
python exp4_oversmoothing_drg.py         | tee log_E4.txt
python exp5_oversmoothing_general.py     | tee log_E5.txt
python exp6_real_graph_gat.py            | tee log_E6.txt
python make_figures.py
