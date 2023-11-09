#ifndef MQTT_COMMON_H
#define MQTT_COMMON_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#ifdef CONFIG_SVC_CV_FLEXIDAG
#include "ambint.h"
#else
#include <stdint.h>
#endif
// COMMON STRUCT

#define SVC_MQTT_C3D_LANE_PIXEL_INTERVAL (3U)
#define SVC_MQTT_C3D_MAX_LANE_POINTS (340U)

typedef struct M3D_MQTT_VEC2F {
  float X;
  float Y;
} __attribute__((packed)) M3D_MQTT_VEC2F;

typedef struct M3D_MQTT_VEC2I {
  int32_t X;
  int32_t Y;
} __attribute__((packed)) M3D_MQTT_VEC2I;

typedef struct M3D_MQTT_VEC3F {
  float X;
  float Y;
  float Z;
} __attribute__((packed)) M3D_MQTT_VEC3F;

typedef struct M3D_MQTT_AABB2F {
  float X;
  float Y;
  float R;
  float B;
} __attribute__((packed)) M3D_MQTT_AABB2F;

typedef struct M3D_MQTT_BBOX3F {
  M3D_MQTT_VEC3F Center;
  M3D_MQTT_VEC3F Extension;
  M3D_MQTT_VEC3F PoseEuler;
} __attribute__((packed)) M3D_MQTT_BBOX3F;

#ifdef __cplusplus
}  // closing brace for extern "C"
#endif  // __cplusplus

#endif  // MQTT_COMMON_H