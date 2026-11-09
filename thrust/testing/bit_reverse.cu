// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/bit_reverse.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <cuda/std/cstdint>

#include <iterator>
#include <vector>

#include <unittest/unittest.h>

template <class Vector>
void test_bit_reverse_copy_subrange()
{
  const Vector input{99, 10, 23, 35, 47, 52, 61, 76, 88, 98};
  Vector output(10, 97);
  const auto end = thrust::bit_reverse_copy(input.begin() + 1, input.end() - 1, output.begin() + 1);
  const Vector expected{97, 10, 52, 35, 76, 23, 61, 47, 88, 97};
  const Vector unchanged{99, 10, 23, 35, 47, 52, 61, 76, 88, 98};
  REQUIRE(output == expected);
  REQUIRE(end == output.end() - 1);
  REQUIRE(input == unchanged);

  Vector restored(8);
  thrust::bit_reverse_copy(output.begin() + 1, output.end() - 1, restored.begin());
  const Vector original(input.begin() + 1, input.end() - 1);
  REQUIRE(restored == original);
}
DECLARE_VECTOR_UNITTEST(test_bit_reverse_copy_subrange);

TEST_CASE("Bit reversal handles power-of-two lengths on host and device", "[bit_reverse]")
{
  const int count = GENERATE(0, 1, 2, 4, 16, 256, 1024);
  thrust::host_vector<int> input(count);
  thrust::host_vector<int> expected(count);
  for (int i = 0; i < count; ++i)
  {
    input[i] = 3 * i + 7;
  }
  for (int i = 0; i < count; ++i)
  {
    // Reverse one bit at a time, independently of the implementation's
    // full-width reversal and leading-zero calculation.
    int index    = i;
    int reversed = 0;
    for (int remaining = count; remaining > 1; remaining /= 2)
    {
      reversed = 2 * reversed + index % 2;
      index /= 2;
    }
    expected[i] = input[reversed];
  }

  thrust::host_vector<int> host_output(count + 2, -1);
  const auto host_end = thrust::bit_reverse_copy(thrust::host, input.begin(), input.end(), host_output.begin() + 1);
  REQUIRE(host_end == host_output.end() - 1);
  REQUIRE(host_output.front() == -1);
  REQUIRE(host_output.back() == -1);
  const thrust::host_vector<int> host_values(host_output.begin() + 1, host_output.end() - 1);
  REQUIRE(host_values == expected);

  const thrust::device_vector<int> device_input = input;
  thrust::device_vector<int> device_output(count + 2, -1);
  const auto device_end =
    thrust::bit_reverse_copy(thrust::device, device_input.begin(), device_input.end(), device_output.begin() + 1);
  REQUIRE(device_end == device_output.end() - 1);
  const thrust::host_vector<int> device_values = device_output;
  REQUIRE(device_values.front() == -1);
  REQUIRE(device_values.back() == -1);
  const thrust::host_vector<int> device_middle(device_values.begin() + 1, device_values.end() - 1);
  REQUIRE(device_middle == expected);
}

TEST_CASE("Bit reversal supports generated input and output-only iterators", "[bit_reverse]")
{
  const thrust::counting_iterator<cuda::std::int64_t> first(100);
  std::vector<cuda::std::int64_t> output;
  thrust::bit_reverse_copy(thrust::seq, first, first + 8, std::back_inserter(output));
  const std::vector<cuda::std::int64_t> expected{100, 104, 102, 106, 101, 105, 103, 107};
  REQUIRE(output == expected);

  const auto discarded = thrust::bit_reverse_copy(thrust::seq, first, first + 8, thrust::make_discard_iterator());
  REQUIRE(discarded == thrust::make_discard_iterator(8));

  // Exercise a 32-bit difference type as well as the default pointer-sized type.
  using Iterator = thrust::counting_iterator<int, thrust::use_default, thrust::use_default, cuda::std::int32_t>;
  thrust::device_vector<int> device_output(8);
  const auto end = thrust::bit_reverse_copy(thrust::device, Iterator(100), Iterator(108), device_output.begin());
  REQUIRE(end == device_output.end());
  const thrust::host_vector<int> actual = device_output;
  const thrust::host_vector<int> expected32{100, 104, 102, 106, 101, 105, 103, 107};
  REQUIRE(actual == expected32);
}
