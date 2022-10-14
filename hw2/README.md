# ADL HW2

### Training 
* Transform Data to swag/squad format
    ```bash=
    : '
    "${1}": path to the context file.
    "${2}": path to the testing file.
    "${3}": path to the output predictions.
    '
    python preprocessing.py --context_path $1 --test_path $2 --train_val True
    ```
* Context Selection
    ```bash=
	mkdir ./ckpt
    bash context_train.sh
    ```
* Question Answering
    ```bash=
    bash qa_train.sh
    ```

### Inference


```bash=
: '
"${1}": path to the context file.
"${2}": path to the testing file.
"${3}": path to the output predictions.
'
bash run.sh "${1}" "${2}" "${3}"
```


### Download Model
```bash=
bash download.sh
```
