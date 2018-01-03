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

#ifdef XORO
// The latest and greatest PRNG, see http://xoroshiro.di.unimi.it
static inline uint64_t rnd(void)
{
    // The state is initialized with the first two outputs of splitmix64
    // seeded with zero (just as suggested in xoroshiro128plus.c).
    static uint64_t s0 = 0xe220a8397b1dcdaf;
    static uint64_t s1 = 0x6e789e6aa1b965f4;
    uint64_t ret = s0 + s1;
    s1 ^= s0;
    s0 = rotl64(s0, 55) ^ s1 ^ (s1 << 14);
    s1 = rotl64(s1, 36);
    return ret;
}
#else
// A faster but otherwise awful PRNG.  It just happens to be good enough
// for the job.  There is a measurable statistical discrepancy with this
// generator replacing a good one, I just don't need that kind of precision
// anyway.  Another reason for using this generator is that its state is
// only a single variable, and so it yields unique numbers (within the period).
// This makes it possible to disable dup detection during simulations.
static inline uint64_t rnd(void)
{
    static uint64_t x = 0xcafef00dd15ea5e5ULL; // PCG_STATE_MCG_64_INITIALIZER
    // Use the previous state as the basis for the return value, to improve
    // instruction-level parallelism.  Rotate the worst 12 bits away into
    // the higher half.  Would like to rotate even more, possibly by 16 bits,
    // but need to keep at least 20 good bits in the higher half to address
    // up to 1M buckets.
    uint64_t ret = rotr64(x, 12);
    // Multiplicative congruential generator with the period of 2^62.
    // The multiplication is executed in parallel even as the caller makes
    // use of "ret" to index into the buckets.
    x = x * 6364136223846793005ULL; // PCG_DEFAULT_MULTIPLIER_64
    return ret;
}
#endif
