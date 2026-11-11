// Native TVM execution. The Rust session serializes every call and retains
// buffers until completion. No C++ exception crosses this private C ABI.
// References: TVM runtime/vm/vm.h, runtime/tensor.h and tvm/ffi/extra/module.h
// from apache-tvm 0.27.0.post1 and apache-tvm-ffi 0.1.15.post2 (Apache-2.0).
#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/extra/module.h>
#include <atomic>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace ffi = tvm::ffi;
using Tensor = ffi::Tensor;
struct Session {
  ffi::Module library, vm;
  ffi::Function call;
  ffi::Function allocate, upload, download;
  DLDevice device;
};
struct View {
  Tensor value;
  std::atomic<size_t> references{1};
  explicit View(Tensor tensor) : value(std::move(tensor)) {}
};

template <class F> int guarded(char* error, size_t capacity, F body) noexcept {
  try {
    body();
    return 0;
  } catch (const std::exception& e) {
    if (capacity) std::snprintf(error, capacity, "%s", e.what());
  } catch (...) {
    if (capacity) std::snprintf(error, capacity, "unknown TVM runtime exception");
  }
  return 1;
}

static DLDataType dtype(uint32_t code) {
  static const DLDataType types[] = {{kDLBool, 8, 1}, {kDLInt, 32, 1}, {kDLInt, 64, 1},
    {kDLUInt, 32, 1}, {kDLUInt, 64, 1}, {kDLFloat, 16, 1}, {kDLBfloat, 16, 1},
    {kDLFloat, 32, 1}, {kDLFloat, 64, 1}};
  if (code >= 9) throw std::invalid_argument("invalid tensor dtype");
  return types[code];
}
static ffi::Function function(const ffi::Module& module, const char* name) {
  auto value = module->GetFunction(name, true);
  if (!value) throw std::invalid_argument(std::string("missing TVM function: ") + name);
  return *value;
}

extern "C" int bauxtvm_load(const char* path, const char* entry, const char* device,
                            void** output, char* error, size_t capacity) {
  *output = nullptr;
  return guarded(error, capacity, [&] {
    if (std::string(device) != "cpu") {
      throw std::invalid_argument("native TVM currently supports CPU");
    }
    auto library = ffi::Module::LoadFromFile(path);
    auto vm = function(library, "vm_load_executable")().cast<ffi::Module>();
    function(vm, "vm_initialization")(int64_t(kDLCPU), int64_t(0), int64_t(2));
    auto call = function(vm, entry);
    auto session = std::make_unique<Session>(Session{
      std::move(library), std::move(vm), std::move(call),
      ffi::Function::GetGlobalRequired("runtime.TVMTensorAllocWithScope"),
      ffi::Function::GetGlobalRequired("runtime.TVMTensorCopyFromBytes"),
      ffi::Function::GetGlobalRequired("runtime.TVMTensorCopyToBytes"), {kDLCPU, 0}});
    *output = session.release();
  });
}
extern "C" void bauxtvm_close(void* context) { delete static_cast<Session*>(context); }
extern "C" void bauxtvm_retain(void* pointer) { ++static_cast<View*>(pointer)->references; }
extern "C" void bauxtvm_release(void* pointer) {
  if (pointer && --static_cast<View*>(pointer)->references == 0) delete static_cast<View*>(pointer);
}
extern "C" int bauxtvm_put(void* context, const size_t* shape, size_t rank, uint32_t kind,
                           const uint8_t* data, size_t length, void** output,
                           char* error, size_t capacity) {
  *output = nullptr;
  return guarded(error, capacity, [&] {
    std::vector<int64_t> dimensions;
    for (size_t i = 0; i < rank; ++i) {
      if (shape[i] > size_t(std::numeric_limits<int64_t>::max())) {
        throw std::invalid_argument("tensor dimension exceeds int64");
      }
      dimensions.push_back(int64_t(shape[i]));
    }
    auto session = static_cast<Session*>(context);
    auto value = session->allocate(ffi::Shape(dimensions), dtype(kind),
                                   session->device, nullptr).cast<Tensor>();
    session->upload(value, const_cast<void*>(static_cast<const void*>(data)), int64_t(length));
    *output = new View(std::move(value));
  });
}
extern "C" int bauxtvm_read(void* context, void* view, uint8_t* output, size_t length,
                            char* error, size_t capacity) {
  return guarded(error, capacity, [&] {
    static_cast<Session*>(context)->download(
      static_cast<View*>(view)->value, static_cast<void*>(output), int64_t(length));
  });
}
extern "C" int bauxtvm_matches(void* pointer, const size_t* shape, size_t rank, uint32_t kind) {
  if (kind >= 9) return 0;
  auto value = static_cast<View*>(pointer)->value.GetDLTensorPtr();
  auto type = dtype(kind);
  if (size_t(value->ndim) != rank || value->dtype.code != type.code ||
      value->dtype.bits != type.bits || value->dtype.lanes != 1) return 0;
  for (size_t i = 0; i < rank; ++i) {
    if (value->shape[i] < 0 || size_t(value->shape[i]) != shape[i]) return 0;
  }
  return 1;
}
extern "C" int bauxtvm_call(void* context, void* const* inputs, size_t count,
                            void** outputs, size_t capacity, char* error, size_t error_capacity) {
  return guarded(error, error_capacity, [&] {
    if (count > size_t(std::numeric_limits<int32_t>::max())) {
      throw std::invalid_argument("too many arguments");
    }
    std::vector<ffi::AnyView> arguments;
    for (size_t i = 0; i < count; ++i) {
      arguments.emplace_back(static_cast<const Tensor&>(static_cast<View*>(inputs[i])->value));
    }
    ffi::Any result;
    static_cast<Session*>(context)->call.CallPacked(arguments.data(), int32_t(count), &result);
    auto values = result.cast<ffi::Array<ffi::Any>>();
    if (values.size() != capacity) {
      throw std::invalid_argument("TVM output count differs from signature");
    }
    std::vector<std::unique_ptr<View>> owned;
    for (auto value : values) {
      owned.push_back(std::make_unique<View>(Tensor(value.cast<Tensor>())));
    }
    for (size_t i = 0; i < capacity; ++i) outputs[i] = owned[i].release();
  });
}
