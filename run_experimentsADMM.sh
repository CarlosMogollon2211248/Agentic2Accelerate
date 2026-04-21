#!/bin/bash

# Definir los valores de sigma que quieres probar
sigmas=(0.02 0.03 0.04 0.05 0.06 0.07 0.08)
# crs=(0.1 0.2 0.3 0.4 0.5 0.7)

# Iterar sobre cada valor
for s in "${sigmas[@]}"
do
   echo "Ejecutando experimento con sigma: $s"
   
   # Creamos un nombre de carpeta único basado en el sigma
   folder="figures_resultsAccADMM/set5_eval_SPC_sigma_$s"
   
   python test_algo_selector_pnp_SPC_ADMM.py \
       --sigma $s\
       --output_dir $folder \
       --rl_checkpoint "results/rl_acc_selector_spcADMM_epoch0009.pth" \
       --max_iter 2000
done