# COMP9417-2022T1
The COMP9417 group project for group Paranormal Distributions

## Developement

### Code structure

```Text
|_ code
    |_ algorithm
    |   |_ main.py
    |   |_ dataset.py
    |   |_ metrics.py
    |   |_ utils.py
    |   |_ modules
    |   |   |_ cnn.py
    |   |   |_ ...     
    |_ input
    |   |_ ...
    |_ output
    |   |_ ...
    |_ ...
```

### How to run

#### Installation
Install all requirements from `requirements.txt`:

```Bash
pip install -r requirements.txt
```

With deafult configuration:
```python 
python main.py
```

or with arguments you want (see `parse_args(args)` in `main.py` for options):

```python 
python main.py <arg1> <arg2> ...
```

### How to develop

WIP