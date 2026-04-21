#!/bin/bash

# Definir los valores de sigma que quieres probar
# sigmas=(0.00001 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1)
crs=(0.1 0.2 0.3 0.4 0.5 0.7)

# Iterar sobre cada valor
for s in "${crs[@]}"
do
   echo "Ejecutando experimento con cr: $s"
   
   # Creamos un nombre de carpeta único basado en el sigma
   folder="figures_resultsAcc/set5_eval_SPC_cr_$s"
   
   python test_algo_selector_pnp_SPC.py \
       --cr $s\
       --sigma 0.0001\
       --output_dir $folder \
       --rl_checkpoint "results/rl_acc_selector_spc.pth" \
       --max_iter 2000
done