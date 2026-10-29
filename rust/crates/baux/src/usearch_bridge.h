#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void* baux_usearch_new(size_t num_dim, char* error, size_t error_capacity) noexcept;
void baux_usearch_destroy(void* handle) noexcept;
bool baux_usearch_reserve(void* handle, size_t capacity, char* error, size_t error_capacity) noexcept;
bool baux_usearch_add(
    void* handle,
    uint64_t key,
    float const* vector,
    size_t num_dim,
    char* error,
    size_t error_capacity) noexcept;
bool baux_usearch_search(
    void const* handle,
    float const* query,
    size_t num_dim,
    size_t wanted,
    bool exact,
    uint64_t* out_keys,
    size_t out_capacity,
    size_t* out_count,
    char* error,
    size_t error_capacity) noexcept;
size_t baux_usearch_expansion_search(void const* handle) noexcept;
size_t baux_usearch_memory_usage(void const* handle) noexcept;

#ifdef __cplusplus
}
#endif
