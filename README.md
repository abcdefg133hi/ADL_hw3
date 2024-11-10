# ADL\_HomeWork III
- Welcome to ADL Homework 3 (The last homework, ya-yeah!!)

## Reproduce
```
./download.sh
./run.sh yourModelPath yourAdapterPath yourInputFilePath yourOutputFilePath
```

## Training
```
python hw3.py --model_name_or_path yourModel --peft_path yourAdapterPath --train_file yourTrainingInput --valid_file yourValidation input --epochs num_epochs --output_file yourOutputFilePath
```
Notice that it won't save your adapter. To solve it, please uncomment line 217 and changed it into your saving path.

## Prediction
```
python hw3.py --model_name_or_path yourModel --peft_path yourAdapterPath --train_file yourTrainingInput --valid_file yourValidation input --output_file yourOutputFilePath --epochs 0
```

## Zero Shot
```
python hw3_zero_shot.py --peft_path yourAdapterPath --train_file yourTrainingInput --valid_file yourValidation input --output_file yourOutputFilePath --epochs 0
```

## Three Shots
```
python hw3_few_shot.py --peft_path yourAdapterPath --train_file yourTrainingInput --valid_file yourValidation input --output_file yourOutputFilePath --epochs 0
```

## Have Fun ^-^
