# Running in dev environment
Activating into virtual environment
  ```
  source venv/bin/activate
  ```

Running NoteBook
  ```
  # Python 2.7
  ipython notebook
  ```

# Installation

Installing Python 2.7
```
brew install python
```

Install PIP
```
wget https://bootstrap.pypa.io/get-pip.py
python get-pip.py
```

Installing virtual environment
  ```
  pip install virtualenv 

  # Run to make sure its been installed
  which virtualenv
  ```

Setting up virtual environment
  ```
  mkdir venv
  virtualenv venv
  ```

Activating into virtual environment
  ```
  source venv/bin/activate
  ```

Install requirements
  ```
  pip install  --index-url=http://pypi.python.org/simple/ --trusted-host pypi.python.org  -r requirements.txt
  ```

Running NoteBook
  ```
  # Python 2.7
  ipython notebook
  ```

Neural networks
  - very resilient against overfitting
  - Dummany classifer = guessing

Normalizing of data between 0 and 1
  - dramatically improves accuracy