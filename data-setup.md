Below are instructions on how to setup the completion3d data loader. New datasets
can be added following the template in shared/datasets/shapenet.py. Feel free to
submit new dataset additions to the benchmark as a pull request.

## Clone Repository

```
git clone git@github.com:lynetcha/completion3d.git
```

## Python Environment

```
cd completion3d
PYTHON_BIN=/path/to/python3.6
virtualenv -p $PYTHON_BIN comp3d_venv
source comp3d_venv/bin/activate
pip install -r requirements/data-requirements.txt

```

## Download Dataset

Download the [completion3D dataset](http://www.completion3D.stanford.edu). You should have the following direcory structure

```
data
├── shapenet
│   ├── synsetoffset2category.txt
│   ├── train
│   ├── val
│   ├── test
│   ├── train.list
│   └── val.list
│   └── test.list
```

## Run Data Loader

```
python shared/datasets/shapenet.py
```
