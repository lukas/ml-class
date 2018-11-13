# Overview

These are materials I use for various classes on deep learning.  Each file is a self contained unit that demonstrates a specific thing.  Downloading or cloning this repository before class is a great way to follow along.

# Reusing the materials

Please feel free to use these materials for your own classes/projects etc.  If you do that, I would love it if you sent me a message and let me know what you're up to.

# Videos

You can find video overviews of a lot of the material at https://youtu.be/Zxrk88rA7fA.

## Prerequisites

These classes are intended for people who are comfortable wirth python.

### Reading material for people who haven't done a lot of programming

If you are uncomfortable opening up a terminal, I strongly recommend doing a quick tutorial before you take this class.  Setting up your machine can be painful but once you're setup you can get a ton out of the class.  I recommend getting started ahead of time.

If you're on Windows I recommend checking out http://thepythonguru.com/.

If you're on a Mac check out http://www.macworld.co.uk/how-to/mac/coding-with-python-on-mac-3635912/

If you're on linux, you're probably already reasonably well setup :).

If you run into trouble, the book Learn Python the Hard Way has installation steps in great detail: https://learnpythonthehardway.org/book/ex0.html.  It also has a refresher on using a terminal in the appendix.

### Reading material for people who are comfortable with programming, but haven't done a lot of python

If you are comfortable opening up a terminal but want a python intro/refresher check out https://www.learnpython.org/ for a really nice introduction to Python.

### Suggestions for people who have done a lot of programming in python

A lot of people like to follow along with ipython or jupyter notebooks and I think that's great!  It makes data exploration easier.  I also really appreciate pull requests to make the code clearer.

If you've never used pandas or numpy - they are great tools and I use them heavily in my work and for this class.  I assume no knlowedge of pandas and numpy but you may want to do some learning on your own.  You can get a quick overview of pandas at http://pandas.pydata.org/pandas-docs/stable/10min.html.  There is a great overview of numpy at https://docs.scipy.org/doc/numpy/user/quickstart.html.

## Installation
I recommend running this code in a pre-configured environment.  You can rent an AWS EC2 node with any of the "Deep Learning" AMIs from aws.amazon.com or a GCP instance.

Once you have a cloud machine setup run:

```
pip install -r requirements.txt
```

You can also install this class locally, but it may be trickier.

### Windows

#### Git

Install git: https://git-scm.com/download/win

#### Anaconda

Install [anaconda](https://repo.continuum.io/archive/Anaconda3-4.4.0-Windows-x86_64.exe)

Try running the following from the command prompt:

```
python --version
```

You should see something like

```
Python 3.6.1 :: Anaconda 4.4.0 (64-bit)
```

If don't see "Anaconda" in the output, search for "anaconda prompt" from the start menu and enter your command prompt this way.   It's also best to use a virtual environment to keep your packages silo'ed.  Do so with:

```
conda create -n ml-class python=3.6
activate ml-class
```

Whenever you start a new terminal, you will need to call `activate ml-class`.

#### Common problems

The most common problem is an old version of python.  Its easy to have multiple versions of python installed at once and Macs in particular come with a default version of python that is too old to install tensorflow.

Try running:

```
python --version
```

If your version is less than 2.7.12, you have a version issue.  Try reinstalling python 2.


#### Clone this github repository
```
git clone https://github.com/lukas/ml-class.git
cd ml-class
```

#### libraries

```
pip install wandb
conda install -c conda-forge scikit-learn
conda install -c conda-forge tensorflow
conda install -c conda-forge keras
```

### Linux and Mac OS X
#### Install python

You can download python from https://www.python.org/downloads/.  There are more detailed instructions for windows installation at https://www.howtogeek.com/197947/how-to-install-python-on-windows/.  

The material should work with python 2 or 3.  *On Windows, you need to install thre 64 bit version of python 3.5 or 3.6 in order to install tensorflow*.

#### Clone this github repository
```
git clone https://github.com/lukas/ml-class.git
cd ml-class
```

If you get an error message here, most likely you don't have git installed.  Go to https://www.atlassian.com/git/tutorials/install-git for intructions on installing git.

#### Install necessary pip libraries
```
pip install -r requirements.txt
```


## Check installation

To make sure your installation works go to the directory where this file is and run
```
python test-scikit.py
```

You should see the output "Scikit is installed!"

```
python test-keras.py
```

You should see the output "Using TensorFlow backend.  Keras is installed!"


