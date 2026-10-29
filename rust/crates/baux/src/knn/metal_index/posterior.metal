#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)

struct Shape {
    uint queries, metrics, input_k, k, skip, has_var, noise, seeds;
    float epistemic, aleatoric;
};

// Same weighted ENN equations as posterior/weighted.rs and CUDA posterior_full.
kernel void enn_moments(device const float* distances [[buffer(0)]],
    device const uint* ids [[buffer(1)]], device const float* outcomes [[buffer(2)]],
    device const float* variances [[buffer(3)]], device const float* scales [[buffer(4)]],
    device float* means [[buffer(5)]], device float* errors [[buffer(6)]],
    device float* weights [[buffer(7)]], device float* l2 [[buffer(8)]],
    device uint* selected [[buffer(9)]], constant Shape& s [[buffer(10)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid>=s.queries*s.metrics) return;
    uint q=gid/s.metrics, m=gid%s.metrics;
    float scale=scales[m], sum=0.0f;
    for(uint j=0;j<s.k;++j) {
        uint at=q*s.input_k+s.skip+j, row=ids[at];
        float variance=s.has_var?variances[row*s.metrics+m]/(scale*scale):0.0f;
        float w=1.0f/max(1e-9f+s.epistemic*distances[at]+s.aleatoric+variance,1e-9f);
        weights[(q*s.k+j)*s.metrics+m]=w;
        if(m==0) selected[q*s.k+j]=row;
        sum+=w;
    }
    float inverse=1.0f/sum, mean=0.0f, squared=0.0f, noise=0.0f;
    for(uint j=0;j<s.k;++j) {
        uint at=(q*s.k+j)*s.metrics+m,row=ids[q*s.input_k+s.skip+j];
        float w=weights[at]*inverse;
        weights[at]=w;
        mean+=w*outcomes[row*s.metrics+m]; squared+=w*w;
        if(s.noise) noise+=w*(s.aleatoric+(s.has_var?variances[row*s.metrics+m]/(scale*scale):0.0f));
    }
    means[gid]=mean;
    errors[gid]=sqrt(max(inverse+noise,1e-9f))*scale;
    l2[gid]=sqrt(squared);
}

inline ulong enn_hash(ulong x) {
    x+=0x9e3779b97f4a7c15UL;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9UL;
    x=(x^(x>>27))*0x94d049bb133111ebUL;
    return x^(x>>31);
}
inline float enn_normal(ulong seed,uint row,uint metric) {
    ulong key=(seed*1000003UL+ulong(row))*1000003UL+ulong(metric);
    float u1=clamp(float(enn_hash(key)>>11)*0x1p-53f,1e-12f,1.0f-1e-7f);
    float u2=float(enn_hash(key^0xD2B74407B1CE6E93UL)>>11)*0x1p-53f;
    return sqrt(-2.0f*log(u1))*cos(6.283185307179586f*u2);
}
kernel void enn_draw(device const float* weights [[buffer(0)]],device const float* l2 [[buffer(1)]],
    device const float* means [[buffer(2)]],device const float* errors [[buffer(3)]],
    device const uint* ids [[buffer(4)]],device const ulong* seeds [[buffer(5)]],
    device float* output [[buffer(6)]],constant Shape& s [[buffer(7)]],uint gid [[thread_position_in_grid]]) {
    uint count=s.queries*s.metrics;
    if(gid>=s.seeds*count) return;
    uint sample=gid/count,at=gid%count,q=at/s.metrics,m=at%s.metrics;
    float sum=0.0f;
    for(uint j=0;j<s.k;++j) sum+=weights[(q*s.k+j)*s.metrics+m]*enn_normal(seeds[sample],ids[q*s.k+j],m);
    output[gid]=means[at]+errors[at]/max(l2[at],1e-12f)*sum;
}
