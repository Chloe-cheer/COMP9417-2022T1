# COMP9417-2022T1
The COMP9417 group project for group Paranormal Distributions

## Developement

### Code structure

```Text
|_ code
    |_ algorithm
    |   |_ main.py
    |   |_ dataset.py
    |   |_ train.py
    |   |_ test.py
    |   |_ utils.py
    |   |_ file_compressor.py
    |   |_ modules
    |   |   |_ cnn.py
    |   |_ experiments
    |   |   |_ ... (NOTE: Consist of all other classifiers we have experimented)
    |   |_ ...     
    |_ input
    |   |_ ... (NOTE: Manually put X_train and X_test here)
    |_ output
    |   |_ ... (NOTE: Default path for saving model and predictions)
    |_ setup.py (NOTE: Default path for saving model and predictions)
```

### How to run

#### Installation

Install all requirements from `requirements.txt`:
```Bash
pip install -r requirements.txt
```

Install packege dependencies:
```Bash
python3 -m pip install -e .
```

### Our Best Model

Run main program with deafult configuration to get best model:
```Bash
python main.py
```
The algorithm we are using for the best model can be found in `algorithm\modules\cnn.py`.

### Other classifiers we trained and experimented

See `algorithm/experiments` directory.