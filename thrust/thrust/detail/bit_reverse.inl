// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#include <thrust/bit_reverse.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/bit_reverse.h>
#include <thrust/system/detail/generic/select_system.h>

THRUST_NAMESPACE_BEGIN

_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy, typename RandomAccessIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator bit_reverse_copy(
  const THRUST_NS_QUALIFIER::detail::execution_policy_base<DerivedPolicy>& exec,
  RandomAccessIterator first,
  RandomAccessIterator last,
  OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::bit_reverse_copy");
  using THRUST_NS_QUALIFIER::system::detail::generic::bit_reverse_copy;
  return bit_reverse_copy(
    THRUST_NS_QUALIFIER::detail::derived_cast(THRUST_NS_QUALIFIER::detail::strip_const(exec)), first, last, result);
}

template <typename RandomAccessIterator, typename OutputIterator>
OutputIterator bit_reverse_copy(RandomAccessIterator first, RandomAccessIterator last, OutputIterator result)
{
  _CCCL_NVTX_RANGE_SCOPE("thrust::bit_reverse_copy");
  using THRUST_NS_QUALIFIER::system::detail::generic::select_system;
  using InputSystem  = typename THRUST_NS_QUALIFIER::iterator_system<RandomAccessIterator>::type;
  using OutputSystem = typename THRUST_NS_QUALIFIER::iterator_system<OutputIterator>::type;
  InputSystem input_system;
  OutputSystem output_system;
  return THRUST_NS_QUALIFIER::bit_reverse_copy(select_system(input_system, output_system), first, last, result);
}

THRUST_NAMESPACE_END
