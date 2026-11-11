// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#include <thrust/system/detail/generic/tag.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename DerivedPolicy, typename RandomAccessIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator bit_reverse_copy(
  THRUST_NS_QUALIFIER::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator first,
  RandomAccessIterator last,
  OutputIterator result);
} // namespace system::detail::generic
THRUST_NAMESPACE_END

#include <thrust/system/detail/generic/bit_reverse.inl>
