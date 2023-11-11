#include "base/io/yaml_util.h"

#include <string>

namespace airos {
namespace base {

bool LoadYamlNodeFromFile(const std::string& yaml_file, YAML::Node* node) {
  if (node == nullptr) return false;
  try {
    *node = YAML::LoadFile(yaml_file);
  } catch (YAML::BadFile& e) {
    LOG_ERROR << "Load " << yaml_file << " with error, YAML::BadFile exception";
    return false;
  } catch (YAML::Exception& e) {
    LOG_ERROR << "Load " << yaml_file
              << " with error, YAML::Exception:" << e.what();
    return false;
  } catch (const std::exception& e) {
    LOG_ERROR << "Load " << yaml_file << " with error: " << e.what();
    return false;
  }
  return true;
}

}  // namespace base
}  // namespace airos
