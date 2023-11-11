    
#pragma once
#include <fcntl.h>

#include <string>

#include "base/common/log.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"

namespace airos {
namespace base {

// @brief load protobuf(TXT) data from file.
template <typename T>
bool ParseProtobufFromFile(const std::string &file_name, T *pb) {
  int fd = open(file_name.c_str(), O_RDONLY);
  if (fd < 0) {
    LOG_ERROR << "ProtobufParser load file failed. file: " << file_name;
    return false;
  }

  google::protobuf::io::FileInputStream fs(fd);
  if (!google::protobuf::TextFormat::Parse(&fs, pb)) {
    LOG_ERROR << "ProtobufParser parse data failed. file:" << file_name;
    close(fd);
    return false;
  }
  close(fd);
  return true;
}

}  // namespace base
}  // namespace airos
