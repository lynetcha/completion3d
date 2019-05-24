PYTHON_DIR=../../comp3d_tf_venv
CUDA_DIR=/usr/local/cuda
cuda_inc=$(CUDA_DIR)/include/
cuda_lib=$(CUDA_DIR)/lib64/
nsync=$(PYTHON_DIR)/lib/python3.6/site-packages/external/nsync/public
nvcc=$(CUDA_DIR)/bin/nvcc
tf_inc=$(PYTHON_DIR)/lib/python3.6/site-packages/tensorflow/include
tf_lib=$(PYTHON_DIR)/lib/python3.6/site-packages/tensorflow

all: tf_nndistance_so.so tf_approxmatch_so.so

tf_nndistance.cu.o: tf_nndistance.cu
	$(nvcc) tf_nndistance.cu -o tf_nndistance.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

tf_nndistance_so.so: tf_nndistance.cpp tf_nndistance.cu.o
	g++ tf_nndistance.cpp tf_nndistance.cu.o -o tf_nndistance_so.so \
	-I $(cuda_inc) -I $(tf_inc) -I $(nsync) \
	-L $(cuda_lib) -lcudart -L $(tf_lib) -ltensorflow_framework \
    -shared -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -fPIC -O2

tf_approxmatch.cu.o: tf_approxmatch.cu
	$(nvcc) tf_approxmatch.cu -o tf_approxmatch.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

tf_approxmatch_so.so: tf_approxmatch.cpp tf_approxmatch.cu.o
	g++ -shared $(CPPFLAGS) tf_approxmatch.cpp tf_approxmatch.cu.o -o tf_approxmatch_so.so \
	-I $(cuda_inc) -I $(tf_inc) -I $(nsync) \
	-L $(cuda_lib) -lcudart -L $(tf_lib) -ltensorflow_framework \
    -shared -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++11 -fPIC -O2

clean:
	rm -rf *.o *.so
