// Private ownership adapter over the pinned public OpenXLA PJRT C API.
// Rust retains the session and serializes buffer access and execution.
#pragma once
#include "pjrt_c_api.h"
#include <atomic>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#define ARGS(type, name) type name{}; name.struct_size = type##_STRUCT_SIZE
extern "C" const PJRT_Api* GetPjrtApi();

inline void discard(const PJRT_Api* api, PJRT_Error* error) noexcept {
  if (!error) return;
  ARGS(PJRT_Error_Destroy_Args, args);
  args.error = error;
  api->PJRT_Error_Destroy(&args);
}
inline void checked(const PJRT_Api* api, PJRT_Error* error) {
  if (!error) return;
  ARGS(PJRT_Error_Message_Args, args);
  args.error = error;
  api->PJRT_Error_Message(&args);
  // Copy the error text before releasing its owning PJRT_Error.
  std::string message;
  try { message.assign(args.message, args.message_size); }
  catch (...) { discard(api, error); throw; }
  discard(api, error);
  throw std::runtime_error(message);
}
template<class F> int guarded(char* error, size_t capacity, F body) noexcept {
  try { body(); return 0; }
  catch (const std::exception& e) {
    if (capacity) std::snprintf(error, capacity, "%s", e.what());
  } catch (...) {
    if (capacity) std::snprintf(error, capacity, "unknown PJRT exception");
  }
  return 1;
}
struct Event {
  const PJRT_Api* api;
  PJRT_Event* value = nullptr;
  explicit Event(const PJRT_Api* owner) : api(owner) {}
  Event(const Event&) = delete;
  Event& operator=(const Event&) = delete;
  ~Event() {
    if (!value) return;
    ARGS(PJRT_Event_Destroy_Args, args);
    args.event = value;
    discard(api, api->PJRT_Event_Destroy(&args));
  }
  void wait() {
    if (!value) throw std::runtime_error("PJRT did not return a completion event");
    ARGS(PJRT_Event_Await_Args, args);
    args.event = value;
    checked(api, api->PJRT_Event_Await(&args));
  }
};
struct Session {
  const PJRT_Api* api;
  PJRT_Client* client = nullptr;
  PJRT_Device* device = nullptr; // Borrowed from client.
  PJRT_LoadedExecutable* executable = nullptr;
  PJRT_Executable* metadata = nullptr;
  size_t outputs = 0;
  explicit Session(const PJRT_Api* owner) : api(owner) {}
  Session(const Session&) = delete;
  Session& operator=(const Session&) = delete;
  ~Session();
};
struct View {
  const PJRT_Api* api;
  PJRT_Buffer* value = nullptr;
  std::atomic<size_t> references{1};
  explicit View(const PJRT_Api* owner) : api(owner) {}
  ~View() {
    if (!value) return;
    ARGS(PJRT_Buffer_Destroy_Args, args);
    args.buffer = value;
    discard(api, api->PJRT_Buffer_Destroy(&args));
  }
};
