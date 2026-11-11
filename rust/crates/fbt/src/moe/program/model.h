#pragma once
#include "portable.h"

// Fixed-shape FBT/PISA MHC4 model. Host inputs are copied during submit;
// returned completion and buffers follow portable.h ownership rules.
typedef struct bauxmodel bauxmodel;
const char* bauxmodel_error(void);
bauxmodel* bauxmodel_create(uint64_t seed);
// backend: 0 = direct native Metal, 1 = TVM Relax with native Metal kernels.
bauxmodel* bauxmodel_createon(uint64_t seed, uint32_t backend);
void bauxmodel_drop(bauxmodel*);
size_t bauxmodel_parameters(const bauxmodel*);
size_t bauxmodel_rows(const bauxmodel*);
bauxpending* bauxmodel_submit(const bauxmodel*, const uint32_t*,
                              const uint32_t*, const uint8_t*, size_t);
