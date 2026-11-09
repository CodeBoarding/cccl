//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "tuning.h"

#include <cub/util_math.cuh>

int nominal_4b_items_to_items(int nominal_4b_items_per_thread, int key_size)
{
  return cub::detail::nominal_4B_items_to_items(nominal_4b_items_per_thread, key_size);
}
