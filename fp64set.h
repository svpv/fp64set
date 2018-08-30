// Copyright (c) 2017, 2018 Alexey Tourbin
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

#pragma once
#include <stdint.h>

#if FP64SET_BENCH
#include <x86intrin.h>
#endif

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// Create a new set of 64-bit fingerprints.  The logsize parameter specifies
// the expected number of elements in the set (e.g. logsize = 10 for 1024).
// Returns NULL on malloc failure.
struct fp64set *fp64set_new(int logsize);
void fp64set_free(struct fp64set *set);

// Exposes only a part of the structure, just enough to inline the calls.
struct fp64set {
    int (*add)(void *set, uint64_t fp);
    bool (*has)(void *set, uint64_t fp);
};

// Add a 64-bit fingerprint to the set.  Returns 0 for a previously added
// fingerprint, 1 when the new fingerprint was added smoothly; 2 if the
// structure has been resized (when this regularly happens more than once,
// it indicates that the initial logsize value passed to fp64set_new was too
// small).  Returns -1 on malloc failure (ENOMEM); or the insertion can fail
// just by chance (EAGAIN), which means that a series of evictions failed,
// and an unrelated fingerprint has been kicked out.  Unless false negatives
// are permitted, the only option in this case is to rebuild the set from
// scratch, fingerprinting the data with a different seed.  The possibility
// of this kind of failure decreases exponentially with logsize.
static inline int fp64set_add(struct fp64set *set, uint64_t fp)
{
#ifdef FP64SET_BENCH
    uint64_t t0 = __rdtsc();
#endif
    int ret = set->add(set, fp);
#ifdef FP64SET_BENCH
    extern uint64_t fp64set_bench_tadd[4];
    extern uint64_t fp64set_bench_nadd[4];
    fp64set_bench_tadd[ret & 3] += __rdtsc() - t0;
    fp64set_bench_nadd[ret & 3] += 1;
#endif
    return ret;
}

// Check if the fingerprint is in the set.
static inline bool fp64set_has(struct fp64set *set, uint64_t fp)
{
#ifdef FP64SET_BENCH
    uint64_t t0 = __rdtsc();
#endif
    bool ret = set->has(set, fp);
#ifdef FP64SET_BENCH
    extern uint64_t fp64set_bench_thas;
    extern uint64_t fp64set_bench_nhas;
    fp64set_bench_thas += __rdtsc() - t0;
    fp64set_bench_nhas += 1;
#endif
    return ret;
}

#ifdef __cplusplus
}
#endif
