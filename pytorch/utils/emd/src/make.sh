set -e
nvcc emd_cuda.cu -o emd_cuda.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
