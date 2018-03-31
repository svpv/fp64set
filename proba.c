// Copyright (c) 2018 Alexey Tourbin
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define FPSET_PROBA
#include "fpset.c"

// CPU intrinsics recognized by gcc.
static inline uint64_t rotl64(uint64_t x, int r) { return (x << r) | (x >> (64 - r)); }
static inline uint64_t rotr64(uint64_t x, int r) { return (x >> r) | (x << (64 - r)); }

// A fast LCG-based PRNG, medium quality.  Just happens to be good enough
// for the job.  Another reason for using this generator is that its state
// is only a single 64-bit variable, and so it yields unique 64-bit numbers.
// This makes it possible to disable dup detection during simulations.
static inline uint64_t rnd(void)
{
    // The state is initialized with the first outputs of splitmix64
    // seeded with zero (much like suggested in xoroshiro128plus.c).
    static uint64_t x = 0xe220a8397b1dcdaf;
    // Use the previous state as the basis for the return value, to improve
    // instruction-level parallelism.  Rotate the worst 12 bits away into
    // the higher half.  Would like to rotate even more, possibly by 16 bits,
    // but need to keep at least 20 good bits in the higher half to address
    // up to 1M buckets.
    uint64_t ret = rotr64(x, 12);
    // The state is updated in parallel even as the caller makes use of "ret"
    // to index into the buckets.  Constants are Knuth's.
    x = x * 0x5851f42d4c957f2d + 0x14057b7ef767814f;
    return ret;
}
