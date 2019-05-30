void approxmatch_cpu(int b,int n,int m,const float * xyz1,
        const float * xyz2, float * match);

int emd_forward(THFloatTensor *xyz1, THFloatTensor *xyz2,
                THFloatTensor *match, THFloatTensor *cost);

int emd_backward(THFloatTensor *xyz1, THFloatTensor *xyz2,
                THFloatTensor *gradxyz1, THFloatTensor *gradxyz2,
                        THFloatTensor * match);

void matchcost_cpu(int b, int n, int m, const float * xyz1,
        const float * xyz2, const float * match, float * cost);

void matchcostgrad_cpu(int b, int n, int m, const float * xyz1,
        const float * xyz2, const float * match, float * grad1, float * grad2);
