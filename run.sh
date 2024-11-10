python3 change_private.py --input_file ${3} --output_file aa.json
python3 hw3.py --model_name_or_path ${1} --peft_path ${2} --train_file aa.json --valid_file aa.json --epochs 0 --output_file ${4}
