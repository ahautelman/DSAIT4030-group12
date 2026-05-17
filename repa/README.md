```
# 1. Establish the baseline performance
python main.py --mode vanilla --output_dir ./results/vanilla

# 2. Evaluate original token-alignment alignment 
python main.py --mode repa --output_dir ./results/repa

# 3. Evaluate spatial-normalization structural alignment
python main.py --mode irepa --output_dir ./results/irepa

# 4. Evaluate Difference-of-Gaussians structural alignment
python main.py --mode dog --output_dir ./results/dog
```