
# Instructions for setting up an ec2 node for deep learning

## Launching node
1) Sign in at https://console.aws.amazon.com/ec2
(Create an account if necessary)

2) Click "Launch Instance"

3) Choose "Ubuntu Server 16.04 LTS (HVM), SSD Volume Type "

4) Choose "p2.xlarge"

5) Choose all defaults but increase disk size to 30GB or more

6) Launch!

You should see a pop up that asks "Select an existing key pair or create a new key pair"

Click "Create new key pair"
For key pair name choose "class"
Download Key Pair

Put the file class.pem in the .ssh directory

Run: `chmod 400 class.pem`

Click "Launch Instances"

## Log in to node

Go to https://console.aws.amazon.com/ec2

Copy the record under "Public DNS" - it should look like ec2-54-163-158-110.compute-1.amazonaws.com

Replace $EC2_HOSTNAME with the record in the following command:

`ssh ubuntu@$EC2_HOSTNAME -i ~/.ssh/class.pem`


## Install

### Basic setup

```
sudo apt-get update
sudo apt-get install build-essential

```

### Install CUDA 8.0

```
wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo dpkg -i cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
rm cuda-repo-ubuntu1604-8-0-local_8.0.44-1_amd64-deb
sudo apt-get update
sudo apt-get install -y cuda
```

### Install cuDNN

Go to https://developer.nvidia.com/cudnn
Click Download

Open Download cuDNN v5.1 for CUDA 8.0
1) Download cuDNN v5.1 Runtime Library for Ubuntu14.04 (Deb)
2) Download cuDNN v5.1 Developer Library for Ubuntu14.04 (Deb)

*Yes you are actually downloading cudnn for ubuntu14.04!*


```
scp -i ~/.ssh/class.pem libcudnn5-dev_5.1.10-1+cuda8.0_amd64.deb ubuntu@$EC2_HOSTNAME:~
scp -i ~/.ssh/class.pem libcudnn5_5.1.10-1+cuda8.0_amd64.deb ubuntu@$EC2_HOSTNAME:~
```

Log back into your ec2 machine with
`ssh ubuntu@$EC2_HOSTNAME -i ~/.ssh/class.pem`

Install libcudnn
```
sudo dpkg -i libcudnn5_5.1.10-1+cuda8.0_amd64.deb
sudo dpkg -i libcudnn5-dev_5.1.10-1+cuda8.0_amd64.deb
```

### Install tensorflow and keras

```
sudo apt install python-pip
pip install --upgrade pip
sudo pip install tensorflow-gpu
sudo pip install keras
```

### Install optional libraries
```
sudo pip install h5py
sudo pip install flask
sudo pip install scikit-image
sudo pip install scipy
sudo pip install pillow
```

```
sudo apt-get install unzip
```

### Configure

Open your .profile file with your favorite text editor
```
nano .profile
```

Add the following lines to the bottom
```
export CUDA_HOME=/usr/local/cuda-8.0
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export PATH=${CUDA_HOME}/bin:${PATH}
```

### Checkout the files for this class!
```
git clone https://github.com/lukas/ml-class
```

### Install Jupyter Notebook
```
sudo apt-get install ipython ipython-notebook
sudo pip install jupyter
```

### Open port on EC2

Go back to https://console.aws.amazon.com/ec2
Right click on the instance created and check the name of the security group
Click "Inbound" and then "Edit"
Add Rule with "Custom TCP" type and Port Range 8888

```
jupyter notebook --generate-config

jupyter notebook password
```

Edit the jupyter notebook config file to allow external connections
```
nano ~/.jupyter/jupyter_notebook_config.py
```

Change the line that says `#c.NotebookApp.ip = ''` to
```
c.NotebookApp.ip = '*'
```

### Start Jupyter Notebook
```
jupyter notebook
```

Note that this is not secure because it connects over http - if you prefer https there are a few more steps
you can follow at http://jupyter-notebook.readthedocs.io/en/latest/public_server.html
