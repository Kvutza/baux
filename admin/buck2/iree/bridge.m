// IREE C runtime wrapper. API and ownership reference:
// https://iree.dev/reference/bindings/c-api/#runtime-api
#include "bridge.h"
#include "iree/runtime/api.h"
#include "iree/hal/drivers/metal/metal_buffer.h"
#include <stdlib.h>
#include <string.h>

struct bauxiree_context {
  iree_runtime_instance_t* instance;
  iree_runtime_session_t* session;
  iree_hal_device_t* device;
  iree_runtime_call_t call;
  int metal;
};

static const iree_hal_element_type_t types[] = {
    IREE_HAL_ELEMENT_TYPE_BOOL_8, IREE_HAL_ELEMENT_TYPE_SINT_32,
    IREE_HAL_ELEMENT_TYPE_SINT_64, IREE_HAL_ELEMENT_TYPE_UINT_32,
    IREE_HAL_ELEMENT_TYPE_UINT_64, IREE_HAL_ELEMENT_TYPE_FLOAT_16,
    IREE_HAL_ELEMENT_TYPE_BFLOAT_16, IREE_HAL_ELEMENT_TYPE_FLOAT_32,
    IREE_HAL_ELEMENT_TYPE_FLOAT_64};

static int finish(iree_status_t status, char* error, size_t capacity) {
  int code = (int)iree_status_code(status);
  if (code) {
    iree_host_size_t length = 0;
    iree_status_format(status, capacity, error, &length);
    if (capacity) error[capacity - 1] = 0;
  }
  iree_status_free(status);
  return code;
}

void bauxiree_close(bauxiree_context* context) {
  if (!context) return;
  if (context->call.session) iree_runtime_call_deinitialize(&context->call);
  iree_runtime_session_release(context->session);
  iree_hal_device_release(context->device);
  iree_runtime_instance_release(context->instance);
  free(context);
}

int bauxiree_open(const char* driver, bauxiree_context** output,
                  char* error, size_t capacity) {
  *output = NULL;
  bauxiree_context* c = calloc(1, sizeof(*c));
  if (!c) return finish(iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED), error, capacity);
  c->metal = strcmp(driver, "metal") == 0;
  iree_allocator_t allocator = iree_allocator_system();
  iree_runtime_instance_options_t options;
  iree_runtime_instance_options_initialize(&options);
  iree_runtime_instance_options_use_all_available_drivers(&options);
  iree_status_t status = iree_runtime_instance_create(&options, allocator, &c->instance);
  if (iree_status_is_ok(status))
    status = iree_runtime_instance_try_create_default_device(c->instance, iree_make_cstring_view(driver), &c->device);
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  if (iree_status_is_ok(status))
    status = iree_runtime_session_create_with_device(c->instance, &session_options, c->device, allocator, &c->session);
  if (iree_status_is_ok(status)) *output = c;
  else bauxiree_close(c);
  return finish(status, error, capacity);
}

int bauxiree_load(const uint8_t* data, size_t length, const char* driver,
                  const char* entry, bauxiree_context** output,
                  char* error, size_t capacity) {
  bauxiree_context* c = NULL;
  int code = bauxiree_open(driver, &c, error, capacity);
  if (code) return code;
  iree_allocator_t allocator = iree_allocator_system();
  iree_status_t status = iree_ok_status();
  void* owned = NULL;
  if (iree_status_is_ok(status)) status = iree_allocator_malloc(allocator, length, &owned);
  if (iree_status_is_ok(status)) {
    memcpy(owned, data, length);
    // append consumes the allocation even when loading fails.
    status = iree_runtime_session_append_bytecode_module_from_memory(c->session, iree_make_const_byte_span(owned, length), allocator);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_initialize_by_name(c->session, iree_make_cstring_view(entry), &c->call);
    // initialize already deinitializes partial call state on failure.
    if (!iree_status_is_ok(status)) memset(&c->call, 0, sizeof(c->call));
  }
  if (iree_status_is_ok(status)) *output = c;
  else bauxiree_close(c);
  return finish(status, error, capacity);
}

int bauxiree_put(bauxiree_context* c, const size_t* shape, size_t rank,
                 uint32_t dtype, const uint8_t* data, size_t length,
                 void** output, char* error, size_t capacity) {
  *output = NULL;
  if (dtype >= sizeof(types) / sizeof(types[0]))
    return finish(iree_make_status(IREE_STATUS_INVALID_ARGUMENT), error, capacity);
  iree_hal_buffer_view_t* view = NULL;
  iree_status_t status = iree_hal_buffer_view_allocate_buffer_copy(
      c->device, iree_hal_device_allocator(c->device), rank, shape, types[dtype],
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                                .usage = IREE_HAL_BUFFER_USAGE_DEFAULT},
      iree_make_const_byte_span(data, length), &view);
  *output = view;
  return finish(status, error, capacity);
}

int bauxiree_read(bauxiree_context* c, void* view, uint8_t* output,
                  size_t length, char* error, size_t capacity) {
  if (iree_hal_buffer_view_byte_length(view) != length)
    return finish(iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "readback byte length differs"), error, capacity);
  return finish(iree_hal_device_transfer_d2h(c->device,
      iree_hal_buffer_view_buffer(view), 0, output, length,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()), error, capacity);
}

void bauxiree_retain(void* view) { iree_hal_buffer_view_retain(view); }

void bauxiree_release(void* view) { iree_hal_buffer_view_release(view); }

int bauxiree_matches(void* view, const size_t* shape, size_t rank, uint32_t dtype) {
  if (dtype >= sizeof(types) / sizeof(types[0]) || iree_hal_buffer_view_shape_rank(view) != rank) return 0;
  const iree_hal_dim_t* actual_shape = iree_hal_buffer_view_shape_dims(view);
  for (size_t i = 0; i < rank; ++i) if (actual_shape[i] != shape[i]) return 0;
  iree_hal_element_type_t actual = iree_hal_buffer_view_element_type(view);
  if ((dtype == 1 || dtype == 3) && actual == IREE_HAL_ELEMENT_TYPE_INT_32) return 1;
  if ((dtype == 2 || dtype == 4) && actual == IREE_HAL_ELEMENT_TYPE_INT_64) return 1;
  return actual == types[dtype];
}

int bauxiree_call(bauxiree_context* c, void* const* inputs, size_t count,
                  void** outputs, size_t capacity, char* error, size_t error_capacity) {
  for (size_t i = 0; i < capacity; ++i) outputs[i] = NULL;
  iree_runtime_call_reset(&c->call);
  iree_status_t status = iree_ok_status();
  for (size_t i = 0; i < count && iree_status_is_ok(status); ++i)
    status = iree_runtime_call_inputs_push_back_buffer_view(&c->call, inputs[i]);
  if (iree_status_is_ok(status)) status = iree_runtime_call_invoke(&c->call, 0);
  if (iree_status_is_ok(status) && iree_vm_list_size(c->call.outputs) != capacity)
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "result count differs");
  for (size_t i = 0; i < capacity && iree_status_is_ok(status); ++i)
    status = iree_runtime_call_outputs_pop_front_buffer_view(&c->call, (iree_hal_buffer_view_t**)&outputs[i]);
  iree_runtime_call_reset(&c->call);
  if (!iree_status_is_ok(status)) {
    for (size_t i = 0; i < capacity; ++i) { bauxiree_release(outputs[i]); outputs[i] = NULL; }
  }
  return finish(status, error, error_capacity);
}

// Import retains MTLBuffer. The caller supplies a lease preventing mutation
// until all views and invocations using this allocation have been released.
int bauxiree_import(bauxiree_context* c, void* handle, size_t offset,
                    size_t length, const size_t* shape, size_t rank,
                    uint32_t dtype, void** output, char* error, size_t capacity) {
  *output = NULL;
  if (!c->metal || !handle || dtype >= sizeof(types) / sizeof(types[0]))
    return finish(iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "Metal import requires a Metal device and valid tensor"), error, capacity);
  id<MTLBuffer> metal = (__bridge id<MTLBuffer>)handle;
  if (offset > metal.length || length > metal.length - offset)
    return finish(iree_make_status(IREE_STATUS_OUT_OF_RANGE, "Metal tensor exceeds allocation"), error, capacity);
  iree_hal_allocator_t* allocator = iree_hal_device_allocator(c->device);
  iree_hal_buffer_t* probe = NULL;
  iree_hal_buffer_params_t params = {.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      .access = IREE_HAL_MEMORY_ACCESS_READ, .usage = IREE_HAL_BUFFER_USAGE_DEFAULT};
  iree_status_t status = iree_hal_allocator_allocate_buffer(allocator, params, 4, &probe);
  if (iree_status_is_ok(status) &&
      iree_hal_metal_buffer_handle(probe).device.registryID != metal.device.registryID)
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "Metal buffer belongs to another device");
  iree_hal_buffer_release(probe);
  iree_hal_external_buffer_t external = {
      .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION, .size = metal.length,
      .handle.device_allocation.ptr = (uint64_t)(uintptr_t)handle};
  iree_hal_buffer_t* buffer = NULL;
  iree_hal_buffer_t* span = NULL;
  if (iree_status_is_ok(status))
    status = iree_hal_allocator_import_buffer(allocator, params, &external,
        iree_hal_buffer_release_callback_null(), &buffer);
  if (iree_status_is_ok(status)) status = iree_hal_buffer_subspan(buffer, offset, length, iree_allocator_system(), &span);
  if (iree_status_is_ok(status))
    status = iree_hal_buffer_view_create(span, rank, shape, types[dtype],
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, iree_allocator_system(), (iree_hal_buffer_view_t**)output);
  iree_hal_buffer_release(span);
  iree_hal_buffer_release(buffer);
  return finish(status, error, capacity);
}

int bauxiree_metal(bauxiree_context* c, void* view, void** handle,
                   uint64_t* offset, char* error, size_t capacity) {
  if (!c->metal) return finish(iree_make_status(IREE_STATUS_INVALID_ARGUMENT, "tensor is not on Metal"), error, capacity);
  iree_hal_buffer_t* buffer = iree_hal_buffer_view_buffer(view);
  *handle = (__bridge void*)iree_hal_metal_buffer_handle(iree_hal_buffer_allocated_buffer(buffer));
  *offset = iree_hal_buffer_byte_offset(buffer);
  return 0;
}
