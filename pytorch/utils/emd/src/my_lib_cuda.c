#include <THC/THC.h>
#include "emd_cuda.h"



extern THCState *state;


int emd_forward_cuda(THCudaTensor *xyz1, THCudaTensor *xyz2,
        THCudaTensor *match, THCudaTensor * cost, THCudaTensor * temp) {
    int success = 0;


    //approxmatchLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,
    //float * match, float * temp);
    success = approxmatchLauncher(xyz1->size[0],
	xyz1->size[1],
	xyz2->size[1],
	THCudaTensor_data(state, xyz1),
	THCudaTensor_data(state, xyz2),
	THCudaTensor_data(state, match),
	THCudaTensor_data(state, temp)
    );

    if (!success) {
    THError("aborting");
    }

    success = 0;
    success = matchcostLauncher(xyz1->size[0],
	xyz1->size[1],
	xyz2->size[1],
	THCudaTensor_data(state, xyz1),
	THCudaTensor_data(state, xyz2),
	THCudaTensor_data(state, match),
	THCudaTensor_data(state, cost)
	);

    if (!success) {
    THError("aborting");
    }
    return 1;
}


int emd_backward_cuda(THCudaTensor *xyz1, THCudaTensor *xyz2, THCudaTensor *gradxyz1,
        THCudaTensor *gradxyz2, THCudaTensor *match) {

    int success = 0;
    success = matchcostgradLauncher(xyz1->size[0],
	xyz1->size[1],
	xyz2->size[1],
	THCudaTensor_data(state, xyz1),
	THCudaTensor_data(state, xyz2),
	THCudaTensor_data(state, match),
	THCudaTensor_data(state, gradxyz1),
	THCudaTensor_data(state, gradxyz2)
	);
	//int NmDistanceGradKernelLauncher(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2, cudaStream_t stream)

    if (!success) {
    THError("aborting");
    }

    return 1;
}



