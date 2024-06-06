# for i in $(seq 10); do 
CUDA_VISIBLE_DEVICES=0 python3 src/main.py --config="vdn" --env-config="hok" with "env_args.map_name=hok"
# done