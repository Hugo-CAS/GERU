Code and datasets for the Graph Evolution Recurrent Unit in our paper "Learning Dynamic Dependencies with Graph Evolution Recurrent Unit for Stock Predictions."

## Environment

Python 3.7 & PyTorch 1.6

## Data

All data, including time series data and sector information, are under the files with dataset names, such as the [rs1000](https://github.com/Hugo-CAS/GERU/tree/master/rs1000) folder.

### Time Series Data

Raw data: files under the [rs1000_rawdata](https://github.com/Hugo-CAS/GERU/tree/master/rs1000/rs1000_rawdata) folder are the historical (6 years) data (i.e., open, high, low, close prices and trading volume) of more than 1,000 stocks traded in US stock market collected from Yahoo Finance.

Processed data: [rs1000_processeddata](https://github.com/Hugo-CAS/GERU/tree/master/rs1000/rs1000_processeddata) is the dataset used to conducted experiments in our paper.

To get the processed data, run the following commands in sequence.
Cleaning data:
```
python clear_data.py
```
Preprocessing data:
```
python preprocessing.py
```
Building dataset:
```
python data.py
```
trade_date.csv includes the calibrated trading dates.

### Sector data

Under data folders, there are sector affiliation files whose names are xxx_list.csv or 电信/工业/xxx .csv.

## Code

### Pre-processing

| Script | Function |
| :-----------: | :-----------: |
| clear_data.py | Clean stock time series data |
| preprocessing.py | Normalize the data and generate the ground truth |
| data.py | Save the data as npy format in the disk|

### Training
| Script | Function |
| :-----------: | :-----------: |
| ./GERU/Trainer.py | Train a model of GERU |
| ./clu_GERU/Trainer.py | Train a model of clu-GERU |


## Run

To repeat the experiment, i.e., train a GERU model, run the following command. 

### GERU
```
python ./GERU/Trainer.py
```

### clu-GERU
```
python ./clu_GERU/Trainer.py
```

### Hyperparameters
All hyperparameters are included in the script Trainer.py. You can set them to test the model performance. The trainer uses an early stopping mechanism to choose the best hyperparameters in the validation set. 


## Cite

If you use the code, please kindly cite the following paper:
```
@article{tian2022learning,
  title={Learning Dynamic Dependencies with Graph Evolution Recurrent Unit for Stock Predictions},
  author={Hu Tian, Xiaolong Zheng, Xingwei Zhang, and Daniel Dajun Zeng},
  year={2022},
  publisher={Under review}
}
```

## Contact

tianhu2018@ia.ac.cn