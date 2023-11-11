#include <string>

#include "base/common/log.h"
#include "yaml-cpp/yaml.h"

namespace airos {
namespace base {

bool LoadYamlNodeFromFile(const std::string &yaml_file, YAML::Node *node);

}
}  // namespace airos
