// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! @brief Copy a power-of-two range in bit-reversed index order.

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/execution_policy.h>

THRUST_NAMESPACE_BEGIN

//! @brief Copy elements in bit-reversed index order using an execution policy.
//!
//! For a range of size N = 2^k, output position i receives the input element
//! whose index is obtained by reversing the low k bits of i. This is the
//! permutation used by radix-2 FFT algorithms; the bits of the values themselves
//! are not changed. An empty range is also supported.
//!
//! For example, eight input elements {10, 11, 12, 13, 14, 15, 16, 17} produce
//! {10, 14, 12, 16, 11, 15, 13, 17}. Applying the permutation twice restores
//! the original order. The index map is computed lazily, without allocating
//! an N-element index array. The selected backend may use temporary storage.
//!
//! @param[in] exec The execution policy controlling parallelization.
//! @param[in] first Beginning of the input range.
//! @param[in] last End of the input range.
//! @param[out] result Beginning of the destination range.
//! @return The output iterator advanced by last - first.
//! @pre The input iterators support random access and last - first is nonnegative.
//! @pre The input length is zero or a power of two.
//! @pre The destination has space for the input length and does not overlap the input.
//! @note This is an out-of-place reordering, not an in-place FFT implementation.
template <typename DerivedPolicy, typename RandomAccessIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator bit_reverse_copy(
  const THRUST_NS_QUALIFIER::detail::execution_policy_base<DerivedPolicy>& exec,
  RandomAccessIterator first,
  RandomAccessIterator last,
  OutputIterator result);

//! @brief Copy elements in bit-reversed index order, selecting the iterator system.
//!
//! This overload has the same ordering and preconditions as the policy overload.
//! The input length must be zero or a power of two, and the input and output
//! ranges must not overlap. Empty and one-element ranges retain their order.
//!
//! @param[in] first Beginning of the random-access input range.
//! @param[in] last End of the input range.
//! @param[out] result Beginning of the destination range.
//! @return The output iterator advanced by last - first.
template <typename RandomAccessIterator, typename OutputIterator>
OutputIterator bit_reverse_copy(RandomAccessIterator first, RandomAccessIterator last, OutputIterator result);

THRUST_NAMESPACE_END

#include <thrust/detail/bit_reverse.inl>
