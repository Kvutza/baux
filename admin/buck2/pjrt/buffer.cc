#include "bridge.h"
#include <limits>

static PJRT_Buffer_Type dtype(uint32_t kind) {
  static const PJRT_Buffer_Type kinds[] = {
    PJRT_Buffer_Type_PRED, PJRT_Buffer_Type_S32, PJRT_Buffer_Type_S64,
    PJRT_Buffer_Type_U32, PJRT_Buffer_Type_U64, PJRT_Buffer_Type_F16,
    PJRT_Buffer_Type_BF16, PJRT_Buffer_Type_F32, PJRT_Buffer_Type_F64};
  if (kind >= 9) throw std::invalid_argument("invalid tensor dtype");
  return kinds[kind];
}
extern "C" int bauxpjrt_put(void* pointer, const size_t* shape, size_t rank, uint32_t kind,
                            const uint8_t* data, size_t length, void** output,
                            char* error, size_t capacity) {
  *output = nullptr;
  return guarded(error, capacity, [&] {
    auto session = static_cast<Session*>(pointer);
    std::vector<int64_t> dimensions;
    static const size_t widths[] = {1, 4, 8, 4, 8, 2, 2, 4, 8};
    size_t bytes = kind < 9 ? widths[kind] : 0;
    auto type = dtype(kind);
    for (size_t i = 0; i < rank; ++i) {
      if (shape[i] > size_t(std::numeric_limits<int64_t>::max()) ||
          (shape[i] && bytes > std::numeric_limits<size_t>::max() / shape[i])) {
        throw std::invalid_argument("tensor dimensions overflow");
      }
      dimensions.push_back(int64_t(shape[i])); bytes *= shape[i];
    }
    if (bytes != length) throw std::invalid_argument("tensor upload byte length differs");
    auto view = std::make_unique<View>(session->api);
    Event copied{session->api};
    ARGS(PJRT_Client_BufferFromHostBuffer_Args, put);
    put.client = session->client; put.data = data; put.type = type;
    put.dims = dimensions.data(); put.num_dims = rank;
    put.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
    put.device = session->device;
    auto status = session->api->PJRT_Client_BufferFromHostBuffer(&put);
    view->value = put.buffer; copied.value = put.done_with_host_buffer;
    checked(session->api, status);
    copied.wait();
    *output = view.release();
  });
}
extern "C" int bauxpjrt_read(void*, void* pointer, uint8_t* output, size_t length,
                             char* error, size_t capacity) {
  return guarded(error, capacity, [&] {
    auto view = static_cast<View*>(pointer);
    ARGS(PJRT_Buffer_Dimensions_Args, shape);
    shape.buffer = view->value;
    checked(view->api, view->api->PJRT_Buffer_Dimensions(&shape));
    std::vector<int64_t> order;
    for (size_t i = shape.num_dims; i > 0; --i) order.push_back(int64_t(i - 1));
    // Compilers may choose a device layout. Host reads always return the
    // dense row-major bytes promised by Buffer::read, including transposes.
    ARGS(PJRT_Buffer_MemoryLayout, layout);
    layout.type = PJRT_Buffer_MemoryLayout_Type_Tiled;
    layout.tiled.struct_size = PJRT_Buffer_MemoryLayout_Tiled_STRUCT_SIZE;
    layout.tiled.minor_to_major = order.data(); layout.tiled.minor_to_major_size = order.size();
    ARGS(PJRT_Buffer_ToHostBuffer_Args, read);
    read.src = view->value; read.host_layout = &layout;
    read.dst = output; read.dst_size = length;
    Event copied{view->api};
    auto status = view->api->PJRT_Buffer_ToHostBuffer(&read);
    copied.value = read.event;
    checked(view->api, status);
    copied.wait();
  });
}
extern "C" int bauxpjrt_matches(void* pointer, const size_t* shape, size_t rank, uint32_t kind) {
  try {
    auto view = static_cast<View*>(pointer);
    ARGS(PJRT_Buffer_Dimensions_Args, dimensions);
    dimensions.buffer = view->value;
    checked(view->api, view->api->PJRT_Buffer_Dimensions(&dimensions));
    ARGS(PJRT_Buffer_ElementType_Args, type);
    type.buffer = view->value;
    checked(view->api, view->api->PJRT_Buffer_ElementType(&type));
    if (type.type != dtype(kind) || dimensions.num_dims != rank) return 0;
    for (size_t i = 0; i < rank; ++i) {
      if (dimensions.dims[i] < 0 || size_t(dimensions.dims[i]) != shape[i]) return 0;
    }
    return 1;
  } catch (...) { return 0; }
}
