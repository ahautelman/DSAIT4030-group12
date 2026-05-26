#!/bin/bash
ARCHS=("sit" "unet")
MODES=("vanilla" "repa" "irepa" "dog")

for arch in "${ARCHS[@]}"; do
  for mode in "${MODES[@]}"; do
    
    # Set appropriate lambda weight
    if [ "$mode" == "repa" ]; then
      LAMBDA=0.4
    elif [ "$mode" == "vanilla" ]; then
      LAMBDA=0.0
    else
      LAMBDA=1.0
    fi
    
    echo "Starting Experiment: Model=$arch | Mode=$mode"
    python main.py \
      --model_type $arch \
      --mode $mode \
      --lambda_repa $LAMBDA \
      --max_steps 100 \
      --batch_size 16 \
      --num_evals 1 \
      --num_eval_images 200 \
      --output_dir ./results/${arch}_${mode}
      
  done
done
