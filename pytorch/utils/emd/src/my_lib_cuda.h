int emd_forward_cuda(THCudaTensor *xyz1, THCudaTensor *xyz2,
                        THCudaTensor *match, THCudaTensor *cost, THCudaTensor * temp);

int emd_backward_cuda(THCudaTensor *xyz1, THCudaTensor *xyz2,
                        THCudaTensor *gradxyz1, THCudaTensor *gradxyz2,
                                                THCudaTensor * match);

