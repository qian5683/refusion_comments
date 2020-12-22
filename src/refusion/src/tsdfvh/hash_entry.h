// Copyright 2019 Emanuele Palazzolo (emanuele.palazzolo@uni-bonn.de), Cyrill Stachniss, University of Bonn
#pragma once

#include "cuda_runtime.h"

#define kFreeEntry -1
#define kLockEntry -2

namespace refusion {

namespace tsdfvh {

/**
 * @brief      Struct that represents a hash entry
 */
struct HashEntry {
  /** Entry position (lower left corner of the voxel block)    输入位置（体素块的左下角）*/ 
  int3 position;
  /** Pointer to the position in the heap of the voxel block  指向体素块堆中位置的指针*/
  int pointer = kFreeEntry;      
};

}  // namespace tsdfvh

}  // namespace refusion
