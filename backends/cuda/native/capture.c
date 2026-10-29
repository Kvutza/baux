/* Optional CUPTI module observer; no kernel rewriting or launch interception.
 * Copy callback-owned cubins before returning. These are observed module images,
 * not a proof of decoder correctness, code execution coverage or GPU semantics.
 */
#define _POSIX_C_SOURCE 200809L
#include <cupti.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static CUpti_SubscriberHandle subscriber;
static const char *directory;
static atomic_ulong sequence;

static void fail(const char *message) {
    fprintf(stderr, "GPU_CAPTURE failure=%s errno=%d\n", message, errno);
    fflush(stderr);
    _Exit(86);
}

static void CUPTIAPI observe(void *userdata, CUpti_CallbackDomain domain,
                            CUpti_CallbackId event, const void *data) {
    (void)userdata;
    if (domain != CUPTI_CB_DOMAIN_RESOURCE || event != CUPTI_CBID_RESOURCE_MODULE_LOADED)
        return;
    const CUpti_ResourceData *resource = data;
    const CUpti_ModuleResourceData *module = resource->resourceDescriptor;
    if (!module || !module->pCubin || !module->cubinSize ||
        module->cubinSize > 256u * 1024u * 1024u)
        fail("invalid-module-image");
    char path[PATH_MAX];
    unsigned long serial = atomic_fetch_add(&sequence, 1);
    int length = snprintf(path, sizeof(path), "%s/module-%ld-%lu-%u.cubin",
                          directory, (long)getpid(), serial, module->moduleId);
    if (length < 0 || (size_t)length >= sizeof(path)) fail("path-too-long");
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) fail("open-image");
    size_t offset = 0;
    while (offset < module->cubinSize) {
        ssize_t written = write(fd, module->pCubin + offset, module->cubinSize - offset);
        if (written < 0 && errno == EINTR) continue;
        if (written <= 0) fail("write-image");
        offset += (size_t)written;
    }
    if (close(fd)) fail("close-image");
    fprintf(stderr, "GPU_CAPTURE module=%u bytes=%zu image=%s\n",
            module->moduleId, module->cubinSize, path);
}

__attribute__((constructor)) static void initialize(void) {
    directory = getenv("BAUX_CUPTI_IMAGES");
    if (!directory || !*directory) return;
    CUptiResult status = cuptiSubscribe(&subscriber, observe, NULL);
    if (status != CUPTI_SUCCESS) fail("subscribe");
    status = cuptiEnableCallback(1, subscriber, CUPTI_CB_DOMAIN_RESOURCE,
                                CUPTI_CBID_RESOURCE_MODULE_LOADED);
    if (status != CUPTI_SUCCESS) fail("enable-module-callback");
    uint32_t version = 0;
    if (cuptiGetVersion(&version) != CUPTI_SUCCESS) fail("version");
    fprintf(stderr, "GPU_CAPTURE cupti=%u pid=%ld\n", version, (long)getpid());
}
