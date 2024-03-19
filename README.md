# COMP9417-2022T1
The COMP9417 group project for group Paranormal Distributions

## Project Disclaimer and Acknowledgment

## Disclaimer

This GitHub repository contains materials developed for educational and demonstration purposes. 

Please be aware that any unauthorized use of these materials, including but not limited to copying, reproducing, or presenting them as one's own work, is strictly prohibited and may constitute academic dishonesty or plagiarism.

As an individual contributor to this project, I have invested time and effort into creating original and innovative work. I expect that others will respect this effort by not engaging in any form of academic dishonesty.


## Usage Guidelines

Educational and Demo Purposes: This repository serves as a demonstration of the project's capabilities and the contributions made by myself, Chloe Chen, and others. You may refer to the materials provided here for learning and understanding concepts related to health check chatbot. However, any direct use of these materials in academic or commercial contexts without proper acknowledgment is prohibited.

Attribution: If you find the content of this repository helpful and wish to reference it in your own projects or research, please provide proper attribution by acknowledging the contributions made by myself, Chloe Chen, and any other contributors mentioned in the project documentation.

No Commercial Use: The materials in this repository are not licensed for commercial use. You may not use them for any commercial purposes without explicit permission from the project contributors.

## Conclusion 

Thank you for taking the time to review this disclaimer. By accessing and using the materials in this repository, you agree to abide by the guidelines outlined above. If you have any questions or concerns regarding the use of these materials, please feel free to contact me directly.




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