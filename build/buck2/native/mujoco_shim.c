#include "gymnasium_assets.h"
#include <mujoco/mujoco.h>
#include <stddef.h>
#include <string.h>
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

static void silent_warning(const char* msg) {
    (void)msg;
}

mjModel* mjshim_load(const char* xml, size_t len, char* err, int err_sz) {
    mju_user_warning = silent_warning;
    mjVFS vfs;
    mj_defaultVFS(&vfs);
    if (mj_addBufferVFS(&vfs, "model.xml", xml, (int)len) != 0) {
        mj_deleteVFS(&vfs);
        return NULL;
    }
    mjModel* m = mj_loadXML("model.xml", &vfs, err, err_sz);
    mj_deleteVFS(&vfs);
    return m;
}

mjModel* mjshim_cheetah(char* err, int err_sz) {
    return mjshim_load(
        GYMNASIUM_HALF_CHEETAH_XML,
        sizeof(GYMNASIUM_HALF_CHEETAH_XML) - 1,
        err,
        err_sz
    );
}

mjModel* mjshim_hopper(char* err, int err_sz) {
    return mjshim_load(
        GYMNASIUM_HOPPER_XML,
        sizeof(GYMNASIUM_HOPPER_XML) - 1,
        err,
        err_sz
    );
}

void mjshim_free(mjModel* m) {
    if (m) {
        mj_deleteModel(m);
    }
}

mjData* mjshim_data(const mjModel* m) {
    return mj_makeData(m);
}

void mjshim_drop(mjData* d) {
    if (d) {
        mj_deleteData(d);
    }
}

void mjshim_reset(const mjModel* m, mjData* d) {
    mj_resetData(m, d);
}

void mjshim_forward(const mjModel* m, mjData* d) {
    mj_forward(m, d);
}

void mjshim_step(const mjModel* m, mjData* d) {
    mj_step(m, d);
}

int mjshim_nq(const mjModel* m) {
    return (int)m->nq;
}

int mjshim_nv(const mjModel* m) {
    return (int)m->nv;
}

int mjshim_nu(const mjModel* m) {
    return (int)m->nu;
}

double mjshim_timestep(const mjModel* m) {
    return (double)m->opt.timestep;
}

const double* mjshim_qpos0(const mjModel* m) {
    return m->qpos0;
}

double* mjshim_qpos(mjData* d) {
    return d->qpos;
}

double* mjshim_qvel(mjData* d) {
    return d->qvel;
}

double* mjshim_ctrl(mjData* d) {
    return d->ctrl;
}

double mjshim_time(const mjData* d) {
    return (double)d->time;
}

void mjshim_gemv(
    int rows,
    int cols,
    const float* weights,
    const float* input,
    const float* bias,
    float* output
) {
    memcpy(output, bias, (size_t)rows * sizeof(float));
#ifdef __APPLE__
    cblas_sgemv(
        CblasRowMajor,
        CblasNoTrans,
        rows,
        cols,
        1.0f,
        weights,
        cols,
        input,
        1,
        1.0f,
        output,
        1
    );
#else
    for (int j = 0; j < rows; ++j) {
        float sum = output[j];
        const float* row = weights + (size_t)j * (size_t)cols;
        for (int i = 0; i < cols; ++i) {
            sum += row[i] * input[i];
        }
        output[j] = sum;
    }
#endif
}

void mjshim_gemvt(
    int rows,
    int cols,
    const float* weights,
    const float* vec,
    float* output
) {
#ifdef __APPLE__
    cblas_sgemv(
        CblasRowMajor,
        CblasTrans,
        rows,
        cols,
        1.0f,
        weights,
        cols,
        vec,
        1,
        0.0f,
        output,
        1
    );
#else
    memset(output, 0, (size_t)cols * sizeof(float));
    for (int j = 0; j < rows; ++j) {
        float vj = vec[j];
        const float* row = weights + (size_t)j * (size_t)cols;
        for (int i = 0; i < cols; ++i) {
            output[i] += row[i] * vj;
        }
    }
#endif
}
