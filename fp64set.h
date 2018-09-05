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

#ifdef __i386__
#define FP64SET_FASTCALL __attribute__((regparm(3)))
#else
#define FP64SET_FASTCALL
#endif

// Exposes only a part of the structure, just enough to inline the calls.
struct fp64set {
    // This guy needs to be 16-byte aligned, have to expose it.
    uint64_t stash[2];
    // Pass fp first, eax:edx may hold hash() return value.
    int (*add)(uint64_t fp, void *set) FP64SET_FASTCALL;
    bool (*del)(uint64_t fp, void *set) FP64SET_FASTCALL;
    // Returns int, let the caller booleanize (can be optimized out).
    int (*has)(uint64_t fp, const void *set) FP64SET_FASTCALL;
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
    return set->add(fp, set);
}

// Delete a fingerprint from a set, returns false if not found.
static inline bool fp64set_del(struct fp64set *set, uint64_t fp)
{
    return set->del(fp, set);
}

// Check if the fingerprint is in the set.
static inline bool fp64set_has(const struct fp64set *set, uint64_t fp)
{
    return set->has(fp, set);
}

#ifdef __cplusplus
}
#endif
