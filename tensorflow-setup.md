Below are instructions on how to setup completion3d training in tensorflow.
New networks can be added following the template in tensorflow/models/TopNet.py 
and adding corresponding import statements in tensorflow/main.py and 
tensorflow/utils/train_utils.py Feel free to submit new model additions to the 
benchmark as a pull request.

## Clone Repository

```
git clone git@github.com:lynetcha/completion3d.git
```

## Install CUDA and CUDNN

Instructions below assume CUDA 9.0 is installed in /usr/local/cuda

```
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

## Tensorflow Python Environment

```
cd completion3d/tensorflow
PYTHON_BIN=/path/to/python3.6
virtualenv -p $PYTHON_BIN comp3d_tf_venv
source comp3d_tf_venv/bin/activate
pip install -r ../requirements/tensorflow-requirements.txt

```

## Build Chamfer and EMD functions

```
cd utils/pc_distance
make
cd ../../../
```

## Run Tensorflow Training/Testing

```
cd tensorflow
```

Link data (see data-setup.md)

```
ln -s /path/to/data data
```

Modify parameters in *run.sh*

```
chmod +x run.sh
./run.sh
```
