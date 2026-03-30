# Rethinking Long-tailed Dataset Distillation: A Uni-Level Framework with Unbiased Recovery and Relabeling
The paper has been accepted as AAAI 2026 oral.

## Expert Model Training
For example, to train on CIFAR-10-LT, run:
```
cd expert
python main.py --dataset cifar10 -a convnet --num_classes 10 --imbanlance_rate 0.01--epochs 200 -b 64 --q 0.8 --gamma1 1
```


## Distilled Image Initialization
To run a specific experiment, use the corresponding script. For example:
```
cd initial
sh scripts/cifar10_10ipc_conv3_to_conv3_cr5.sh
```

## Unbiased Recovery
### CIFAR-10-LT
Adjust the paths and hyperparameters in  ```recover_cifar10/recover.sh``` and ```recover_cifar10/recover.py```, then run:
```
cd recover_cifar10
sh recover.sh
```

### CIFAR-100-LT
Adjust the paths and hyperparameters in  ```recover_cifar100/recover.sh``` and ```recover_cifar100/recover.py```, then run:
```
cd recover_cifar100
sh recover.sh
```

### Tiny-ImageNet-LT
Change imbalance factor in ```recover_tiny/tiny_in_dataset.py```, adjust the paths and hyperparameters in  ```recover_tiny/recover.sh``` and ```recover_tiny/recover.py``` , then run:
```
cd recover_tiny
sh recover.sh
```

### ImageNet-LT
Adjust the paths and hyperparameters in  ```recover_1k/recover.sh``` and ```recover_1k/recover.py```, then run:
```
cd recover_1k
sh recover.sh
```

## Unbiased Relabeling and Student Training 
### CIFAR-10-LT
Adjust the paths and hyperparameters in  ```train_cifar10/train.sh``` and ```train_cifar10/direct_train.py```, then run:
```
cd train_cifar10
sh train.sh
```

### CIFAR-100-LT
Adjust the paths and hyperparameters in  ```train_cifar100/train.sh``` and ```train_cifar100/direct_train.py```, then run:
```
cd train_cifar100
sh train.sh
```

### Tiny-ImageNet-LT
Change imbalance factor in train_tiny/tiny_in_dataset.py, adjust the paths and hyperparameters in  ```train_tiny/train.sh``` and ```train_tiny/direct_train.py```, then run:
```
cd train_tiny
sh train.sh
```

### ImageNet-LT
Adjust the paths and hyperparameters in  ```train_1k/train.sh``` and ```train_1k/direct_train.py```, then run:
```
cd train_1k
sh train.sh
```
