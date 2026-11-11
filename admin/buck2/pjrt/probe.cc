// Exercise the public PJRT C ABI directly, with the exact matching XLA header.
// The CPU plugin is OpenXLA's implementation; cyxla only packages its entry point.
// Reference: https://openxla.org/xla/pjrt/c-api
#include "pjrt_c_api.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" const PJRT_Api* GetPjrtApi();
static const PJRT_Api* api;
#define ARGS(type, name) type name{}; name.struct_size = type##_STRUCT_SIZE

static void require(bool condition, const char* message) {
  if (!condition) { std::fprintf(stderr, "%s\n", message); std::exit(1); }
}
static void check(PJRT_Error* error) {
  if (!error) return;
  ARGS(PJRT_Error_Message_Args, message);
  message.error = error;
  api->PJRT_Error_Message(&message);
  std::fprintf(stderr, "%.*s\n", int(message.message_size), message.message);
  ARGS(PJRT_Error_Destroy_Args, destroy);
  destroy.error = error;
  api->PJRT_Error_Destroy(&destroy);
  std::exit(1);
}
static void wait(PJRT_Event* event) {
  ARGS(PJRT_Event_Await_Args, ready);
  ready.event = event;
  check(api->PJRT_Event_Await(&ready));
  ARGS(PJRT_Event_Destroy_Args, destroy);
  destroy.event = event;
  check(api->PJRT_Event_Destroy(&destroy));
}
static void release(PJRT_Buffer* buffer) {
  ARGS(PJRT_Buffer_Destroy_Args, destroy);
  destroy.buffer = buffer;
  check(api->PJRT_Buffer_Destroy(&destroy));
}

int main() {
  api = GetPjrtApi();
  require(api && api->struct_size >= PJRT_Api_STRUCT_SIZE, "PJRT table is too short");
  require(api->pjrt_api_version.major_version == PJRT_API_MAJOR &&
          api->pjrt_api_version.minor_version == PJRT_API_MINOR, "PJRT header version mismatch");
  ARGS(PJRT_Plugin_Initialize_Args, initialize);
  check(api->PJRT_Plugin_Initialize(&initialize));
  ARGS(PJRT_Client_Create_Args, create);
  check(api->PJRT_Client_Create(&create));
  ARGS(PJRT_Client_PlatformName_Args, platform);
  platform.client = create.client;
  check(api->PJRT_Client_PlatformName(&platform));
  require(platform.platform_name_size == 3 &&
          std::memcmp(platform.platform_name, "cpu", 3) == 0, "expected CPU plugin");
  ARGS(PJRT_Client_AddressableDevices_Args, devices);
  devices.client = create.client;
  check(api->PJRT_Client_AddressableDevices(&devices));
  require(devices.num_addressable_devices >= 1, "no CPU device");

  char source[] = R"(module {
    func.func @main(%x: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
      %a = stablehlo.add %x, %x : tensor<4xf32>
      %b = stablehlo.multiply %a, %a : tensor<4xf32>
      return %a, %b : tensor<4xf32>, tensor<4xf32>
    }
  })";
  ARGS(PJRT_Program, program);
  program.code = source; program.code_size = sizeof(source) - 1;
  program.format = "mlir"; program.format_size = 4;
  ARGS(PJRT_Client_Compile_Args, compile);
  compile.client = create.client; compile.program = &program;
  // CompileOptionsProto.executable_build_options (field 3), with one replica
  // (field 4) and one partition (field 5). Empty protobuf options mean zero
  // replicas and abort inside this XLA release, rather than returning an error.
  // xla/pjrt/proto/compile_options.proto at the pinned OpenXLA revision.
  const char compile_options[] = {0x1a, 4, 0x20, 1, 0x28, 1};
  compile.compile_options = compile_options;
  compile.compile_options_size = sizeof(compile_options);
  check(api->PJRT_Client_Compile(&compile));

  float input[] = {1, 2, 3, 4};
  int64_t shape[] = {4};
  ARGS(PJRT_Client_BufferFromHostBuffer_Args, upload);
  upload.client = create.client; upload.data = input;
  upload.type = PJRT_Buffer_Type_F32; upload.dims = shape; upload.num_dims = 1;
  upload.host_buffer_semantics = PJRT_HostBufferSemantics_kImmutableOnlyDuringCall;
  upload.device = devices.addressable_devices[0];
  check(api->PJRT_Client_BufferFromHostBuffer(&upload));
  wait(upload.done_with_host_buffer);
  input[0] = 99; // Upload must own its contents independently of host mutation.
  PJRT_Buffer* arguments[] = {upload.buffer};
  PJRT_Buffer** argument_lists[] = {arguments};
  PJRT_Buffer* outputs[2] = {};
  PJRT_Buffer** output_lists[] = {outputs};
  PJRT_Event* events[1] = {};
  int64_t preserve[] = {0};
  ARGS(PJRT_ExecuteOptions, options);
  options.non_donatable_input_indices = preserve;
  options.num_non_donatable_input_indices = 1;
  ARGS(PJRT_LoadedExecutable_Execute_Args, execute);
  execute.executable = compile.executable; execute.options = &options;
  execute.argument_lists = argument_lists; execute.num_devices = 1; execute.num_args = 1;
  execute.output_lists = output_lists; execute.device_complete_events = events;
  check(api->PJRT_LoadedExecutable_Execute(&execute));
  wait(events[0]);
  for (int i = 0; i < 2; ++i) {
    float output[4] = {};
    ARGS(PJRT_Buffer_ToHostBuffer_Args, read);
    read.src = outputs[i]; read.dst = output; read.dst_size = sizeof(output);
    check(api->PJRT_Buffer_ToHostBuffer(&read));
    wait(read.event);
    for (int j = 0; j < 4; ++j) {
      float value = 2 * (j + 1);
      require(output[j] == (i ? value * value : value), "native result differs");
    }
    release(outputs[i]);
  }
  release(upload.buffer);
  ARGS(PJRT_LoadedExecutable_Destroy_Args, executable);
  executable.executable = compile.executable;
  check(api->PJRT_LoadedExecutable_Destroy(&executable));
  ARGS(PJRT_Client_Destroy_Args, client);
  client.client = create.client;
  check(api->PJRT_Client_Destroy(&client));
  std::printf("PJRT %d.%d: native XLA CPU compilation, execution, events and owned upload passed; no Metal device\n",
              PJRT_API_MAJOR, PJRT_API_MINOR);
}
