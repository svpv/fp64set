// Copyright (c) 2017 Alexey Tourbin
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

#ifndef FPSET_H
#define FPSET_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

// Create a new set of fingerprints.  The logsize parameter specifies the
// expected number of elements in the set (e.g. logsize = 10 for 1024).
// Returns NULL on malloc failure.
struct fpset *fpset_new(int logsize);
void fpset_free(struct fpset *set);

// Add a 64-bit fingerprint to the set.  Returns -1 on malloc failure.
// Returns 0 when the element was added smoothly.  Returns 1 when the
// internal structure was reallocated (if this happens more than a couple
// of times, this indicates that the logsize parameter passed to fpset_new
// was too small).  Returns 2 if the internal structure has been switched
// from Cuckoo filter to the ordered set of numbers.
int fpset_add(struct fpset *set, uint64_t fp) __attribute((nonnull));

// Check if the fingerprint is in the set.
bool fpset_has(struct fpset *set, uint64_t fp) __attribute((nonnull));

#ifdef __cplusplus
}
#endif
#endif
