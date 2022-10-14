python preprocessing.py --context_path $1 --test_path $2 
python inference.py --data_dir ./mydata --output_dir ./result --context_path $1 --test_path $2 --output_file $3 --ckpt_dir ./ckpt