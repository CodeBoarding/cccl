// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <thrust/bit_reverse.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

#include <cassert>
#include <iostream>

// Radix-2 FFTs commonly reorder samples by reversing the bits of their indices.
// For eight samples, index 001 becomes 100, 011 becomes 110, and so on.
// This example performs only that permutation, not the FFT butterflies.
int main()
{
  const thrust::host_vector<int> samples{10, 11, 12, 13, 14, 15, 16, 17};
  const thrust::device_vector<int> input = samples;
  thrust::device_vector<int> reordered(input.size());

  thrust::bit_reverse_copy(thrust::device, input.begin(), input.end(), reordered.begin());

  const thrust::host_vector<int> actual = reordered;
  const thrust::host_vector<int> expected{10, 14, 12, 16, 11, 15, 13, 17};
  assert(actual == expected);
  std::cout << "Bit-reversed samples:";
  for (const auto sample : actual)
  {
    std::cout << ' ' << sample;
  }
  std::cout << '\n';

  // The permutation is its own inverse. Use a separate destination: overlapping
  // input and output are not supported by bit_reverse_copy.
  thrust::device_vector<int> restored(input.size());
  thrust::bit_reverse_copy(reordered.begin(), reordered.end(), restored.begin());
  const thrust::host_vector<int> round_trip = restored;
  assert(round_trip == samples);
  std::cout << "Original order restored\n";
}
