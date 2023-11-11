#ifndef CYBER_TIME_RATE_H_
#define CYBER_TIME_RATE_H_

#include "cyber/time/duration.h"
#include "cyber/time/time.h"

namespace apollo {
namespace cyber {

class Rate {
 public:
  explicit Rate(double frequency);
  explicit Rate(uint64_t nanoseconds);
  explicit Rate(const Duration&);
  void Sleep();
  void Reset();
  Duration CycleTime() const;
  Duration ExpectedCycleTime() const { return expected_cycle_time_; }

 private:
  Time start_;
  Duration expected_cycle_time_;
  Duration actual_cycle_time_;
};

}  // namespace cyber
}  // namespace apollo

#endif  // CYBER_TIME_RATE_H_
