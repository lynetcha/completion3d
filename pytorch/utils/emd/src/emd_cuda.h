#ifdef __cplusplus
extern "C" {
#endif

//int NmDistanceKernelLauncher(int b,int n,const float * xyz,int m,const float * xyz2,float * result,int * result_i,float * result2,int * result2_i, cudaStream_t stream);
//int NmDistanceGradKernelLauncher(int b,int n,const float * xyz1,int m,const float * xyz2,const float * grad_dist1,const int * idx1,const float * grad_dist2,const int * idx2,float * grad_xyz1,float * grad_xyz2, cudaStream_t stream);

int matchcostgradLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * grad1,float * grad2);
int matchcostLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,const float * match,float * out);
int approxmatchLauncher(int b,int n,int m,const float * xyz1,const float * xyz2,float * match, float * temp);


#ifdef __cplusplus
}
#endif
