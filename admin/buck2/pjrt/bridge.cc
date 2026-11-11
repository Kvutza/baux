#include "bridge.h"

static const PJRT_Api* initialize() {
  static const PJRT_Api* api = [] {
    auto value = GetPjrtApi();
    if (!value || value->struct_size < PJRT_Api_STRUCT_SIZE ||
        value->pjrt_api_version.major_version != PJRT_API_MAJOR ||
        value->pjrt_api_version.minor_version != PJRT_API_MINOR) {
      throw std::runtime_error("PJRT plugin does not match the pinned C API header");
    }
    ARGS(PJRT_Plugin_Initialize_Args, args);
    checked(value, value->PJRT_Plugin_Initialize(&args));
    return value;
  }();
  return api;
}
Session::~Session() {
  ARGS(PJRT_Executable_Destroy_Args, meta);
  meta.executable = metadata;
  discard(api, api->PJRT_Executable_Destroy(&meta));
  ARGS(PJRT_LoadedExecutable_Destroy_Args, loaded);
  loaded.executable = executable;
  discard(api, api->PJRT_LoadedExecutable_Destroy(&loaded));
  ARGS(PJRT_Client_Destroy_Args, owner);
  owner.client = client;
  discard(api, api->PJRT_Client_Destroy(&owner));
}

extern "C" int bauxpjrt_compile(const char* source, size_t length, const char* device,
                                const char* options, size_t options_length, size_t input_count,
                                void** output, char* error, size_t capacity) {
  *output = nullptr;
  return guarded(error, capacity, [&] {
    if (std::string(device) != "cpu") {
      throw std::invalid_argument("the bundled XLA PJRT plugin supports CPU only");
    }
    auto api = initialize();
    auto session = std::make_unique<Session>(api);
    ARGS(PJRT_Client_Create_Args, create);
    auto status = api->PJRT_Client_Create(&create);
    session->client = create.client;
    checked(api, status);
    ARGS(PJRT_Client_AddressableDevices_Args, devices);
    devices.client = session->client;
    checked(api, api->PJRT_Client_AddressableDevices(&devices));
    if (!devices.num_addressable_devices) throw std::runtime_error("PJRT has no CPU device");
    session->device = devices.addressable_devices[0];
    ARGS(PJRT_Program, program);
    program.code = const_cast<char*>(source); program.code_size = length;
    program.format = "mlir"; program.format_size = 4;
    // Private ABI: Rust encodes validated input ShapeProtos and nonzero replica
    // and partition counts. No arbitrary user-supplied options reach this call.
    ARGS(PJRT_Client_Compile_Args, compile);
    compile.client = session->client; compile.program = &program;
    compile.compile_options = options; compile.compile_options_size = options_length;
    status = api->PJRT_Client_Compile(&compile);
    session->executable = compile.executable;
    checked(api, status);
    ARGS(PJRT_LoadedExecutable_GetExecutable_Args, metadata);
    metadata.loaded_executable = session->executable;
    status = api->PJRT_LoadedExecutable_GetExecutable(&metadata);
    session->metadata = metadata.executable;
    checked(api, status);
    ARGS(PJRT_Executable_NumOutputs_Args, outputs);
    outputs.executable = session->metadata;
    checked(api, api->PJRT_Executable_NumOutputs(&outputs));
    session->outputs = outputs.num_outputs;
    ARGS(PJRT_Executable_ParameterMemoryKinds_Args, parameters);
    parameters.executable = session->metadata;
    checked(api, api->PJRT_Executable_ParameterMemoryKinds(&parameters));
    if (parameters.num_parameters != input_count) {
      throw std::invalid_argument("PJRT input count differs from signature");
    }
    *output = session.release();
  });
}
extern "C" void bauxpjrt_close(void* pointer) { delete static_cast<Session*>(pointer); }
extern "C" void bauxpjrt_retain(void* pointer) { ++static_cast<View*>(pointer)->references; }
extern "C" void bauxpjrt_release(void* pointer) {
  if (pointer && --static_cast<View*>(pointer)->references == 0) delete static_cast<View*>(pointer);
}

extern "C" int bauxpjrt_call(void* pointer, void* const* inputs, size_t count,
                             void** outputs, size_t capacity, char* error, size_t error_capacity) {
  return guarded(error, error_capacity, [&] {
    auto session = static_cast<Session*>(pointer);
    auto api = session->api;
    if (session->outputs != capacity) throw std::invalid_argument("PJRT output count differs from signature");
    std::vector<PJRT_Buffer*> arguments, results(capacity, nullptr);
    std::vector<int64_t> preserve;
    std::vector<std::unique_ptr<View>> views;
    for (size_t i = 0; i < count; ++i) {
      arguments.push_back(static_cast<View*>(inputs[i])->value);
      preserve.push_back(int64_t(i));
    }
    // Allocate all owning wrappers before launching native work. Even an
    // asynchronous error must release every returned buffer and event.
    for (size_t i = 0; i < capacity; ++i) views.push_back(std::make_unique<View>(api));
    PJRT_Buffer* const* argument_lists[] = {arguments.data()};
    PJRT_Buffer** output_lists[] = {results.data()};
    Event completion{api};
    ARGS(PJRT_ExecuteOptions, options);
    options.non_donatable_input_indices = preserve.data();
    options.num_non_donatable_input_indices = preserve.size();
    ARGS(PJRT_LoadedExecutable_Execute_Args, execute);
    execute.executable = session->executable; execute.options = &options;
    execute.argument_lists = argument_lists; execute.num_devices = 1; execute.num_args = count;
    execute.output_lists = output_lists; execute.device_complete_events = &completion.value;
    auto status = api->PJRT_LoadedExecutable_Execute(&execute);
    for (size_t i = 0; i < capacity; ++i) views[i]->value = results[i];
    checked(api, status);
    completion.wait();
    for (size_t i = 0; i < capacity; ++i) outputs[i] = views[i].release();
  });
}
