#pragma once
#include <stddef.h>
#include <stdint.h>

// Private ABI. All handles and data ranges must be live for each call.
// The Rust wrapper serializes session access and retains buffer owners.
typedef struct bauxiree_context bauxiree_context;
int bauxiree_load(const uint8_t*, size_t, const char*, const char*,
                  bauxiree_context**, char*, size_t);
void bauxiree_close(bauxiree_context*);
int bauxiree_put(bauxiree_context*, const size_t*, size_t, uint32_t,
                 const uint8_t*, size_t, void**, char*, size_t);
int bauxiree_read(bauxiree_context*, void*, uint8_t*, size_t, char*, size_t);
void bauxiree_release(void*);
int bauxiree_call(bauxiree_context*, void* const*, size_t, void**, size_t, char*, size_t);
int bauxiree_matches(void*, const size_t*, size_t, uint32_t);

int bauxiree_open(const char*, bauxiree_context**, char*, size_t);
int bauxiree_import(bauxiree_context*, void*, size_t, size_t, const size_t*, size_t,
                   uint32_t, void**, char*, size_t);
int bauxiree_metal(bauxiree_context*, void*, void**, uint64_t*, char*, size_t);

void bauxiree_retain(void*);
