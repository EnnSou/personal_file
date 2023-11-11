#pragma once

#include <map>
#include <string>
#include <vector>

#include "base/common/log.h"

namespace airos {
namespace base {

// idea from boost any but make it more simple and don't use type_info.
class Any {
 public:
  Any() : content_(NULL) {}

  template <typename ValueType>
  explicit Any(const ValueType &value)
      : content_(new Holder<ValueType>(value)) {}

  Any(const Any &other)
      : content_(other.content_ ? other.content_->Clone() : nullptr) {}

  ~Any() { delete content_; }

  template <typename ValueType>
  ValueType *AnyCast() {
    return content_ ? &(static_cast<Holder<ValueType> *>(content_)->held_)
                    : nullptr;
  }

 private:
  class PlaceHolder {
   public:
    virtual ~PlaceHolder() {}
    virtual PlaceHolder *Clone() const = 0;
  };

  template <typename ValueType>
  class Holder : public PlaceHolder {
   public:
    explicit Holder(const ValueType &value) : held_(value) {}
    virtual ~Holder() {}
    virtual PlaceHolder *Clone() const { return new Holder(held_); }

    ValueType held_;
  };

  PlaceHolder *content_;
};

class ObjectFactory {
 public:
  ObjectFactory() {}
  virtual ~ObjectFactory() {}
  virtual Any NewInstance() { return Any(); }
  ObjectFactory(const ObjectFactory &) = delete;
  ObjectFactory &operator=(const ObjectFactory &) = delete;

 private:
};

typedef std::map<std::string, ObjectFactory *> FactoryMap;
typedef std::map<std::string, FactoryMap> BaseClassMap;
BaseClassMap &GlobalFactoryMap();

bool GetRegisteredClasses(
    const std::string &base_class_name,
    std::vector<std::string> *registered_derived_classes_names);

}  // namespace base
}  // namespace airos

#define PERCEPTION_REGISTER_REGISTERER(base_class)                      \
  class base_class##Registerer {                                        \
    typedef ::airos::base::Any Any;                                     \
    typedef ::airos::base::FactoryMap FactoryMap;                       \
                                                                        \
   public:                                                              \
    static base_class *GetInstanceByName(const ::std::string &name) {   \
      FactoryMap &map = ::airos::base::GlobalFactoryMap()[#base_class]; \
      FactoryMap::iterator iter = map.find(name);                       \
      if (iter == map.end()) {                                          \
        for (auto c : map) {                                            \
          LOG_ERROR << "Instance:" << c.first;                          \
        }                                                               \
        LOG_ERROR << "Get instance " << name << " failed.";             \
        return nullptr;                                                 \
      }                                                                 \
      Any object = iter->second->NewInstance();                         \
      return *(object.AnyCast<base_class *>());                         \
    }                                                                   \
    static std::vector<base_class *> GetAllInstances() {                \
      std::vector<base_class *> instances;                              \
      FactoryMap &map = ::airos::base::GlobalFactoryMap()[#base_class]; \
      instances.reserve(map.size());                                    \
      for (auto item : map) {                                           \
        Any object = item.second->NewInstance();                        \
        instances.push_back(*(object.AnyCast<base_class *>()));         \
      }                                                                 \
      return instances;                                                 \
    }                                                                   \
    static const ::std::string GetUniqInstanceName() {                  \
      FactoryMap &map = ::airos::base::GlobalFactoryMap()[#base_class]; \
      CHECK_EQ(map.size(), 1U) << map.size();                           \
      return map.begin()->first;                                        \
    }                                                                   \
    static base_class *GetUniqInstance() {                              \
      FactoryMap &map = ::airos::base::GlobalFactoryMap()[#base_class]; \
      CHECK_EQ(map.size(), 1U) << map.size();                           \
      Any object = map.begin()->second->NewInstance();                  \
      return *(object.AnyCast<base_class *>());                         \
    }                                                                   \
    static bool IsValid(const ::std::string &name) {                    \
      FactoryMap &map = ::airos::base::GlobalFactoryMap()[#base_class]; \
      return map.find(name) != map.end();                               \
    }                                                                   \
  };

#define PERCEPTION_REGISTER_CLASS(clazz, name)                                \
  namespace {                                                                 \
  class ObjectFactory##name : public airos::base::ObjectFactory {             \
   public:                                                                    \
    virtual ~ObjectFactory##name() {}                                         \
    virtual ::airos::base::Any NewInstance() {                                \
      return ::airos::base::Any(new name());                                  \
    }                                                                         \
  };                                                                          \
  __attribute__((constructor)) void RegisterFactory##name() {                 \
    ::airos::base::FactoryMap &map =                                          \
        ::airos::base::GlobalFactoryMap()[#clazz];                            \
    if (map.find(#name) == map.end()) map[#name] = new ObjectFactory##name(); \
  }                                                                           \
  }  // namespace
