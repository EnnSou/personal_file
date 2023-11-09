#ifndef MQTTIDL_0_8_7_1_H
#define MQTTIDL_0_8_7_1_H

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#include "./mqtt_common.h"

// MQTT DATA STRUCT v0.8.7.1

typedef struct M3D_MQTT_OBJECT_SCORED_CATEGORY_V0_8_7_1 {
  uint16_t Category;
  float Confidence;
  uint16_t Reserved_;
} __attribute__((packed)) M3D_MQTT_OBJECT_SCORED_CATEGORY_V0_8_7_1;

typedef struct M3D_MQTT_GROUND_CONTACT_V0_8_7_1 {
  uint16_t Type;
  float Confidence;
  M3D_MQTT_VEC2F Pt2fImg;
  M3D_MQTT_VEC2F GcpLaneOffsetPix;
  uint16_t Reserved_;
} __attribute__((packed)) M3D_MQTT_GROUND_CONTACT_V0_8_7_1;

#define M3D_MQTT_OBJECT_SCORED_CATEGORY_LENGTH_V0_8_7_1 (7U)
#define M3D_MQTT_GROUND_CONTACT_LENGTH_V0_8_7_1 (4U)
typedef struct M3D_MQTT_OBJECT_RESULT_V0_8_7_1 {
  uint8_t CategoryLength;
  M3D_MQTT_OBJECT_SCORED_CATEGORY_V0_8_7_1
  ScoredCategories[M3D_MQTT_OBJECT_SCORED_CATEGORY_LENGTH_V0_8_7_1];

  // 2D
  M3D_MQTT_AABB2F Box2fImg;

  // 3D
  M3D_MQTT_VEC2F AmodelCenter;
  M3D_MQTT_BBOX3F Box3fCcs;
  float LatDistanceCcsM;

  // Other attributes
  uint8_t GroundContactLength;
  M3D_MQTT_GROUND_CONTACT_V0_8_7_1
  GroundContacts[M3D_MQTT_GROUND_CONTACT_LENGTH_V0_8_7_1];
  uint16_t Orientation;

  // Uncertainty
  float DepUncertaintyMm;
  float RotYUncertaintyCcsRr;
  M3D_MQTT_VEC3F DimUncertaintyMm;

  // Flags
  uint8_t IsCipv;
  uint8_t IsTruncated;
  uint8_t IsOnEgoRoad;
  uint8_t IsGeneralObstacle;
} __attribute__((packed)) M3D_MQTT_OBJECT_RESULT_V0_8_7_1;

#define M3D_MQTT_LANE_LINE_PTS_LENGTH_V0_8_7_1 (340U)
typedef struct M3D_MQTT_LANE_LINE_RESULT_V0_8_7_1 {
  uint16_t LaneLineId;
  uint16_t LaneLineType;
  uint16_t LaneLineColor;
  float Confidence;
  uint16_t PtLength;
  M3D_MQTT_VEC2F Pts2fImg[M3D_MQTT_LANE_LINE_PTS_LENGTH_V0_8_7_1];
  float PtsConfidence[M3D_MQTT_LANE_LINE_PTS_LENGTH_V0_8_7_1];
} __attribute__((packed)) M3D_MQTT_LANE_LINE_RESULT_V0_8_7_1;

#define M3D_MQTT_CONFIG_LENGTH_V0_8_7_1 (100U)
#define M3D_MQTT_DIAGNOSTIC_LENGTH_V0_8_7_1 (50U)
#define M3D_MQTT_OBJECT_LENGTH_V0_8_7_1 (100U)
#define M3D_MQTT_LANE_LINE_LENGTH_V0_8_7_1 (10U)
typedef struct M3D_MQTT_FRAME_RESULT_V0_8_7_1 {
  uint16_t MsgCode = 0;
  uint32_t MsgSize = 0;
  uint64_t CaptureTimeNs = 0;
  uint32_t FrameNum = 0;

  //
  // Configs
  //
  // Number of configs used
  uint8_t ConfigLength;
  // Pre-allocate 100 placeholders for configs, manual alignment needed
  uint8_t Configs[M3D_MQTT_CONFIG_LENGTH_V0_8_7_1];

  // Diagnostics
  uint8_t DiagnosticLength;
  float Diagnostics[M3D_MQTT_DIAGNOSTIC_LENGTH_V0_8_7_1];

  // Lengths for each struct
  uint16_t ObjectLength = 0;
  M3D_MQTT_OBJECT_RESULT_V0_8_7_1 Objects[M3D_MQTT_OBJECT_LENGTH_V0_8_7_1];
  uint16_t LaneLineLength;
  M3D_MQTT_LANE_LINE_RESULT_V0_8_7_1
  LaneLines[M3D_MQTT_LANE_LINE_LENGTH_V0_8_7_1];

  uint16_t MsgEnd;
} __attribute__((packed)) M3D_MQTT_FRAME_RESULT_V0_8_7_1;

#define M3D_MQTT_MSG_CODE_V0_8_7_1 (7U)

// ::common::utils::mqtt

typedef struct M3D_MQTT_CONFIGS_V0_8_7_1 {
  uint8_t ConfigHasLateralDistance;
  uint8_t ConfigHasDirectRegRoty;
  uint8_t ConfigHasGcpLaneOffset;
  uint8_t ConfigHasLaneType;
  uint8_t ConfigHasLaneColor;
  uint8_t ConfigHasUndistortedAmodelCenter;
  uint8_t ConfigHasDepUncertainty;
  uint8_t ConfigGcpBboxExtensionRatio;
  uint8_t ConfigSceneLightingCategoryLength;
  uint8_t ConfigSceneWeatherCategoryLength;
  uint8_t ConfigGcpBboxCeilingRatio;
  uint8_t ConfigLaneInvisibleConfidenceOffset;
  // next 10 position for lane uuid
  uint8_t ConfigLaneUUID[10];
  // next 10 position for fork lane keypoint num
  uint8_t ConfigForkKpsNum[10];
} __attribute__((packed)) M3D_MQTT_CONFIGS_V0_8_7_1;

#define V0_8_7_1_M3D_MQTT_CONFIG_UUID_START (12U)
#define V0_8_7_1_M3D_MQTT_CONFIG_FORK_PTS_START (22U)
#define V0_8_7_1_M3D_MQTT_CONFIG_LENGTH (32U)

#ifdef __cplusplus
}  // closing brace for extern "C"
#endif  // __cplusplus

#endif  // MQTTIDL_0_8_7_1_H