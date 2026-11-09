// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/permutation_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/system/detail/generic/bit_reverse.h>

#include <cuda/std/__bit/bit_reverse.h>
#include <cuda/std/__bit/countl.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/make_unsigned.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
template <typename Index>
struct bit_reverse_index
{
  int shift;

  [[nodiscard]] _CCCL_HOST_DEVICE constexpr Index operator()(Index index) const noexcept
  {
    using Unsigned = ::cuda::std::make_unsigned_t<Index>;
    return static_cast<Index>(::cuda::std::bit_reverse(static_cast<Unsigned>(index)) >> shift);
  }
};

template <typename DerivedPolicy, typename RandomAccessIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator bit_reverse_copy(
  THRUST_NS_QUALIFIER::execution_policy<DerivedPolicy>& exec,
  RandomAccessIterator first,
  RandomAccessIterator last,
  OutputIterator result)
{
  using Index       = typename ::cuda::std::iterator_traits<RandomAccessIterator>::difference_type;
  using Unsigned    = ::cuda::std::make_unsigned_t<Index>;
  const Index count = last - first;
  _CCCL_ASSERT(count >= 0, "bit_reverse_copy requires a nonnegative range length");
  if (count == 0)
  {
    return result;
  }
  _CCCL_ASSERT(::cuda::std::has_single_bit(static_cast<Unsigned>(count)),
               "bit_reverse_copy requires a power-of-two range length");

  // A singleton has a zero-bit index. Use a zero shift rather than shifting
  // by the full word width; its only index is zero either way.
  const int shift = count == 1 ? 0 : ::cuda::std::countl_zero(static_cast<Unsigned>(count)) + 1;
  const auto map  = THRUST_NS_QUALIFIER::make_transform_iterator(
    THRUST_NS_QUALIFIER::counting_iterator<Index>(0), bit_reverse_index<Index>{shift});
  return THRUST_NS_QUALIFIER::copy(
    exec,
    THRUST_NS_QUALIFIER::make_permutation_iterator(first, map),
    THRUST_NS_QUALIFIER::make_permutation_iterator(first, map + count),
    result);
}
} // namespace system::detail::generic
THRUST_NAMESPACE_END
