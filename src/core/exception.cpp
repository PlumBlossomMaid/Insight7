// src/core/exception.cpp
#include "insight/c_api/exception.h"
#include <string>

static thread_local std::string last_error_message;

#ifdef __cplusplus
extern "C" {
#endif

void insight_set_last_error(const char *msg) {
  last_error_message = msg ? msg : "";
}

const char *insight_get_last_error() { return last_error_message.c_str(); }

#ifdef __cplusplus
}
#endif