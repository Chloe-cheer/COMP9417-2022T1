# COMP9417-2022T1
The COMP9417 group project for group Paranormal Distributions

## Developement

### Code structure

```Text
|_ code
    |_ algorithm
    |   |_ main.py
    |   |_ dataset.py
    |   |_ modules
    |   |   |_ cnn.py
    |   |_ ...     
    |_ input
    |   |_ ...
    |_ output
    |   |_ ...
    |_ setup.py
    |_ notebook
        |_ ...
```

### How to run

#### Installation
Install all requirements from `requirements.txt`:

```Bash
pip install -r requirements.txt
```

Install packeges:

```Bash
python3 -m pip install -e .
```

### Our Best Model

Run main program with deafult configuration to get best model:
```Bash
python main.py
```


### Other classifier we trained and experimented

See `notebooks/`.

#### Balanced Classes
balanced_classes.ipynb

#### Haralick
haralick.ipynb