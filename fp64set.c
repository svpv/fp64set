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

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include "fp64set.h"

// The real fp64set structure (upconverted from "void *set").
struct set {
    // To reduce the failure rate, one or two fingerprints can be stashed.
    // When only one fingerprint is stashed, we have stash[0] == stash[1].
    uint64_t stash[2];
    // Virtual functions.
    int (*add)(uint64_t fp, void *set) FP64SET_FASTCALL;
    int (*del)(uint64_t fp, void *set) FP64SET_FASTCALL;
    int (*has)(uint64_t fp, void *set) FP64SET_FASTCALL;
    // The number of buckets - 1, helps indexing into the buckets.
    size_t mask;
    // The buckets (malloc'd); each bucket has bsize slots.
    // Two-dimensional structure is emulated with pointer arithmetic.
    uint64_t *bb;
    // The total number of unique fingerprints added to buckets,
    // not including the stashed fingerprints.
    size_t cnt;
    // The number of fingerprints stashed: 0, 1, or 2.
    int nstash;
    // The number of buckets, the logarithm: 4..32.
    int logsize;
    // The number of slots in each bucket: 2, 3, or 4.
    int bsize;
};

// Make two indexes out of a fingerprint.
// Fingerprints are treated as two 32-bit hash values for this purpose.
#define Hash1(fp, mask) ((fp >> 00) & mask)
#define Hash2(fp, mask) ((fp >> 32) & mask)
#define FP2I(fp, mask)		\
    i1 = Hash1(fp, mask);	\
    i2 = Hash2(fp, mask)
// Further identify the buckets.
#define FP2IB(fp, bb, mask)	\
    FP2I(fp, mask);		\
    b1 = bb + bsize * i1;	\
    b2 = bb + bsize * i2
// Further declare vars.
#define dFP2IB(fp, bb, mask)	\
    size_t i1, i2;		\
    uint64_t *b1, *b2;		\
    FP2IB(fp, bb, mask)

#define unlikely(cond) __builtin_expect(cond, 0)
#define HIDDEN __attribute__((visibility("hidden")))

// The inline functions below rely heavily on constant propagation.
#define inline inline __attribute__((always_inline))

// Check if a fingerprint has already been inserted.  Note that only two memory
// locations are accessed (which translates into only two cache lines); this is
// one reason why fp64set_has() is 2-3 times faster than std::unordered_set<uint64_t>::find().
// Further, I prefer to beget a completely branchless binary code, which
// works faster with real data: although equality sometimes holds, if the CPU
// tries to bet on when and where the equality holds, it loses.  (Branching,
// on the other hand, might work faster if the branches are completely
// predictable, e.g. when running Monte Carlo simulations to estimate the
// probability of failure - in this case, fingerprints are all different and
// equality almost never holds.  However, just in this case, it is possible
// to use a full-period PRNG with 64-bit state and avoid the check entirely.)
static inline int has(uint64_t fp, uint64_t *b1, uint64_t *b2,
	bool nstash, uint64_t *stash, int bsize)
{
    // Issue loads for both buckets.
    int has1 = fp == b1[0];
    int has2 = fp == b2[0];
    // Stashed elements can be checked in the meantime.
    if (nstash) {
	has1 |= fp == stash[0];
	has2 |= fp == stash[1];
    }
    // Check the rest of the slots.
    if (bsize > 1) {
	has1 |= fp == b1[1];
	has2 |= fp == b2[1];
    }
    if (bsize > 2) {
	has1 |= fp == b1[2];
	has2 |= fp == b2[2];
    }
    if (bsize > 3) {
	has1 |= fp == b1[3];
	has2 |= fp == b2[3];
    }
    return has1 | has2;
}

// With older compilers, pass -DFP64SET_SSE4=0.
#ifndef FP64SET_SSE4
#if defined(__i386__) || defined(__x86_64__) || \
    defined(__SSE4_1__) || defined(__AVX__)
#define FP64SET_SSE4 1
#else
#define FP64SET_SSE4 0
#endif
#endif

#if FP64SET_SSE4
#include <smmintrin.h>
#define SSE4_FUNC __attribute__((target("sse4.1")))
#endif

// On 64-bit systems, assume malloc'd chunks are aligned to 16 bytes.
// This should help to elicit aligned SSE2 instructions.
// On i686, malloc aligns to 16 bytes since glibc-2.26~173.
#if SIZE_MAX > UINT32_MAX || defined(__x86_64__) || \
   (defined(__i386__) && 100*__GLIBC__+__GLIBC_MINOR__ >= 226)
#define A16(p) __builtin_assume_aligned(p, 16)
#else
#define A16(p) __builtin_assume_aligned(p, 8)
// Disable SSE2 aligned intrinsics.
#define _mm_load_si128 _mm_loadu_si128
#endif

// A version optimized for SSE4.1, uses _mm_cmpeq_epi64.
#if FP64SET_SSE4
static inline SSE4_FUNC int has_sse4(uint64_t fp, uint64_t *b1, uint64_t *b2,
	bool nstash, uint64_t *stash, int bsize)
{
    __m128i xmm7;
    if (bsize == 3) {
	__m128i xmm1 = _mm_loadu_si128((__m128i *) b1);
	__m128i xmm2 = _mm_loadu_si128((__m128i *) b2);
	__m128i xmm0 = _mm_set1_epi64x(fp);
	__m128i xmm3 = _mm_set_epi64x(b1[2], b2[2]);
	if (nstash)
	    xmm7 = _mm_load_si128((__m128i *) stash);
	xmm1 = _mm_cmpeq_epi64(xmm1, xmm0);
	xmm2 = _mm_cmpeq_epi64(xmm2, xmm0);
	xmm3 = _mm_cmpeq_epi64(xmm3, xmm0);
	if (nstash) {
	    xmm7 = _mm_cmpeq_epi64(xmm7, xmm0);
	    xmm1 = _mm_or_si128(xmm1, xmm2);
	    xmm3 = _mm_or_si128(xmm3, xmm7);
	    xmm1 = _mm_or_si128(xmm1, xmm3);
	    return _mm_movemask_epi8(xmm1);
	}
	xmm1 = _mm_or_si128(xmm1, xmm2);
	xmm1 = _mm_or_si128(xmm1, xmm3);
	return _mm_movemask_epi8(xmm1);
    }
    __m128i xmm1 = _mm_load_si128((__m128i *) b1);
    __m128i xmm2 = _mm_load_si128((__m128i *) b2);
    __m128i xmm0 = _mm_set1_epi64x(fp);
    if (nstash)
	xmm7 = _mm_load_si128((__m128i *) stash);
    if (bsize == 2) {
	xmm1 = _mm_cmpeq_epi64(xmm1, xmm0);
	xmm2 = _mm_cmpeq_epi64(xmm2, xmm0);
	if (nstash) {
	    xmm7 = _mm_cmpeq_epi64(xmm7, xmm0);
	    xmm1 = _mm_or_si128(xmm1, xmm2);
	    xmm1 = _mm_or_si128(xmm1, xmm7);
	    return _mm_movemask_epi8(xmm1);
	}
	xmm1 = _mm_or_si128(xmm1, xmm2);
	return _mm_movemask_epi8(xmm1);
    }
    __m128i xmm3 = _mm_load_si128((__m128i *) b1 + 1);
    __m128i xmm4 = _mm_load_si128((__m128i *) b2 + 1);
    xmm1 = _mm_cmpeq_epi64(xmm1, xmm0);
    xmm2 = _mm_cmpeq_epi64(xmm2, xmm0);
    if (nstash)
	xmm7 = _mm_cmpeq_epi64(xmm7, xmm0);
    xmm3 = _mm_cmpeq_epi64(xmm3, xmm0);
    xmm4 = _mm_cmpeq_epi64(xmm4, xmm0);
    xmm1 = _mm_or_si128(xmm1, xmm2);
    if (nstash) {
	xmm3 = _mm_or_si128(xmm3, xmm7);
	xmm1 = _mm_or_si128(xmm1, xmm4);
	xmm1 = _mm_or_si128(xmm1, xmm3);
	return _mm_movemask_epi8(xmm1);
    }
    xmm3 = _mm_or_si128(xmm3, xmm4);
    xmm1 = _mm_or_si128(xmm1, xmm3);
    return _mm_movemask_epi8(xmm1);
}
#endif

// Template for set->has virtual functions.
static inline int t_has(struct set *set, uint64_t fp, bool nstash, int bsize)
{
    dFP2IB(fp, set->bb, set->mask);
    return has(fp, b1, b2, nstash, set->stash, bsize);
}

#if FP64SET_SSE4
static inline SSE4_FUNC int t_has_sse4(struct set *set, uint64_t fp, bool nstash, int bsize)
{
    dFP2IB(fp, set->bb, set->mask);
    return has_sse4(fp, b1, b2, nstash, set->stash, bsize);
}
#endif

// Instantiate generic functions, only prototypes for now.
#define VFUNC(NAME) \
    FP64SET_FASTCALL \
    int fp64set_##NAME(uint64_t fp, void *set)
#define MakeVFuncs(NB, ST)  \
    static VFUNC(add##NB##st##ST); \
    static VFUNC(del##NB##st##ST); \
    static VFUNC(has##NB##st##ST);
#define MakeAllVFuncs	\
    MakeVFuncs(2, 0)	\
    MakeVFuncs(2, 1)	\
    MakeVFuncs(3, 0)	\
    MakeVFuncs(3, 1)	\
    MakeVFuncs(4, 0)	\
    MakeVFuncs(4, 1)
MakeAllVFuncs

// Instantiate sse4 functions, only prototypes for now.
#if FP64SET_SSE4
#undef MakeVFuncs
#define MakeVFuncs(NB, ST)		    \
    static VFUNC(add##NB##st##ST##sse4) SSE4_FUNC; \
    HIDDEN VFUNC(has##NB##st##ST##sse4) SSE4_FUNC;
MakeAllVFuncs
#endif

// How to initialize vtab slots.
#if FP64SET_SSE4
#define cpu_supports_sse4 __builtin_cpu_supports("sse4.1")
#define SetVFuncs(set, NB, ST, SSE4)			\
do {							\
    if (SSE4) {						\
	set->add = fp64set_add##NB##st##ST##sse4;	\
	set->has = fp64set_has##NB##st##ST##sse4;	\
    }							\
    else {						\
	set->add = fp64set_add##NB##st##ST;		\
	set->has = fp64set_has##NB##st##ST;		\
    }							\
    set->del = fp64set_del##NB##st##ST;			\
} while (0)
#else
#define cpu_supports_sse4 0
#define SetVFuncs(set, NB, ST, SSE4)			\
do {							\
    set->add = fp64set_add##NB##st##ST;			\
    set->del = fp64set_del##NB##st##ST;			\
    set->has = fp64set_has##NB##st##ST;			\
} while (0)
#endif
// In case NB is not a literal.
#define SelectVFuncs(set, NB, ST, SSE4)			\
do {							\
    if (NB == 2)					\
	SetVFuncs(set, 2, ST, SSE4);			\
    else if (NB == 3)					\
	SetVFuncs(set, 3, ST, SSE4);			\
    else						\
	SetVFuncs(set, 4, ST, SSE4);			\
} while (0)

// The sentinels at the end of bb[], for fp64set_next.
#define SENTINELS 3

struct fp64set *fp64set_new(int logsize)
{
    assert(logsize >= 0);
    if (logsize < 4)
	logsize = 4;
    // The limit on 32-bit platforms is 2GB, logsize=28 allocates 4GB.
    if (logsize > 27 && sizeof(size_t) < 5)
	return errno = ENOMEM, NULL;
    // The ultimate limit: two 32-bit halves out of each fingerprint.
    if (logsize > 32)
	return errno = E2BIG, NULL;

    // Starting with two slots per bucket.
    size_t nb = (size_t) 1 << logsize;
    uint64_t *bb = calloc(2 * nb + SENTINELS, sizeof(uint64_t));
    if (!bb)
	return NULL;

    // The blank value for bb[0][*] slots is UINT64_MAX.
    memset(A16(bb), 0xff, 2 * sizeof(uint64_t));
    memset(A16(bb + 2 * nb), 0xff, SENTINELS * sizeof(uint64_t));

    struct set *set = malloc(sizeof *set);
    if (!set)
	return free(bb), NULL;

    SetVFuncs(set, 2, 0, cpu_supports_sse4);

    set->mask = nb - 1;
    set->bb = bb;
    set->stash[0] = set->stash[1] = 0;
    set->cnt = 0;
    set->nstash = 0;
    set->logsize = logsize;
    set->bsize = 2;

    return (struct fp64set *) set;
}

// Test if a fingerprint at bb[i][*] is actually a free slot.
// Note that a bucket can only keep hold of such fingerprints that hash
// into the bucket.  This obviates the need for separate bookkeeping.
static inline bool freeSlot(uint64_t fp, size_t i)
{
    // Slots must be initialized to 0, except that
    // bb[0][*] slots must be initialized to UINT64_MAX aka -1.
    return fp == 0 - (i == 0);
}

#if FP64SET_DEBUG > 1
#include <stdio.h>
#include <inttypes.h>
#include <t1ha.h>
#endif

void fp64set_free(struct fp64set *arg)
{
    struct set *set = (void *) arg;
    if (!set)
	return;
#ifdef FP64SET_DEBUG
    // The number of fingerprints must match the occupied slots.
    size_t cnt = 0;
    size_t mask = set->mask;
    size_t bsize = set->bsize;
    for (size_t i = 0; i <= mask; i++) {
	uint64_t *b = set->bb + bsize * i;
	for (int j = 0; j < bsize; j++) {
	    uint64_t fp = b[j];
	    if (freeSlot(fp, i))
		continue;
	    size_t i1 = Hash1(fp, mask);
	    size_t i2 = Hash2(fp, mask);
	    assert(i == i1 || i == i2);
	    cnt++;
	}
    }
    assert(set->cnt == cnt);
#endif
    // Hash the elements in the buckets and in the stash.
    // Useful when you optimize the code and it runs suspiciously fast. :-)
#if FP64SET_DEBUG > 1
    uint64_t hash = t1ha(set->stash, sizeof set->stash, set->nstash);
    hash = t1ha(set->bb, sizeof(uint64_t) * bsize * (mask + 1), hash);
    fprintf(stderr, "%s logsize=%d bsize=%d nstash=%d cnt=%zu hash=%016" PRIx64 "\n",
	    __func__, set->logsize, set->bsize, set->nstash, cnt, hash);
#endif
    free(set->bb);
    free(set);
}

// That much one needs to know upon the first reading.
// The reset is fp64set_add() stuff.

// Add an element to either of its buckets, preferably to the least loaded.
static inline bool justAdd2(uint64_t fp, uint64_t *b1, size_t i1, uint64_t *b2, size_t i2, int bsize)
{
#if defined(__i386__)
    // Precalculate freeSlot() values, faster due to register pressure.
    uint64_t blank1 = 0 - (i1 == 0);
    uint64_t blank2 = 0 - (i2 == 0);
    if (b1[0] == blank1) return b1[0] = fp, true;
    if (b2[0] == blank2) return b2[0] = fp, true;
    if (b1[1] == blank1) return b1[1] = fp, true;
    if (b2[1] == blank2) return b2[1] = fp, true;
    if (bsize > 2) if (b1[2] == blank1) return b1[2] = fp, true;
    if (bsize > 2) if (b2[2] == blank2) return b2[2] = fp, true;
    if (bsize > 3) if (b1[3] == blank1) return b1[3] = fp, true;
    if (bsize > 3) if (b2[3] == blank2) return b2[3] = fp, true;
#else
    // Otherwise I've got one more trick up in my sleeve: after the buckets
    // are initialized, we have b[0] == b[1], and so on.  When a fingerprint
    // is placed into b[0], the equality breaks.  In other words, b[j] is
    // a free slot iff b[j] == b[j+1].  This works for all but the last slot.
    if (b1[0] == b1[1]) return b1[0] = fp, true;
    if (b2[0] == b2[1]) return b2[0] = fp, true;
    if (bsize > 2) if (b1[1] == b1[2]) return b1[1] = fp, true;
    if (bsize > 2) if (b2[1] == b2[2]) return b2[1] = fp, true;
    if (bsize > 3) if (b1[2] == b1[3]) return b1[2] = fp, true;
    if (bsize > 3) if (b2[2] == b2[3]) return b2[2] = fp, true;
    // When adding to the last slot in a bucket, need to use freeSlot.
    if (freeSlot(b1[bsize-1], i1)) return b1[bsize-1] = fp, true;
    if (freeSlot(b2[bsize-1], i2)) return b2[bsize-1] = fp, true;
#endif
    return false;
}

// Add an element to one bucket (because the other is known to be full).
static inline bool justAdd1(uint64_t fp, uint64_t *b, size_t i, int bsize)
{
#if defined(__i386__)
    uint64_t blank = 0 - (i == 0);
    if (b[0] == blank) return b[0] = fp, true;
    if (b[1] == blank) return b[1] = fp, true;
    if (bsize > 2) if (b[2] == blank) return b[2] = fp, true;
    if (bsize > 3) if (b[3] == blank) return b[3] = fp, true;
#else
    if (b[0] == b[1]) return b[0] = fp, true;
    if (bsize > 2) if (b[1] == b[2]) return b[1] = fp, true;
    if (bsize > 3) if (b[2] == b[3]) return b[2] = fp, true;
    if (freeSlot(b[bsize-1], i)) return b[bsize-1] = fp, true;
#endif
    return false;
}

// When all slots for a fingerprint are occupied, insertion "kicks out"
// an already existing fingerprint and tries to place it into the alternative
// slot, thus triggering a series of evictions.  Returns false with the
// kicked-out fingerprint in ofp.
static inline bool kickAdd(uint64_t fp, uint64_t *bb, uint64_t *b, size_t i,
	uint64_t *ofp, int logsize, size_t mask, int bsize)
{
    int maxkick = logsize << 1;
    do {
	// Put at the top, kick out from the bottom.
	// Using *ofp as a temporary register.
	*ofp = b[0];
	b[0] = b[1];
	if (bsize > 2) b[1] = b[2];
	if (bsize > 3) b[2] = b[3];
	b[bsize-1] = fp, fp = *ofp;
	// Ponder over the fingerprint that's been kicked out.
	// Find out the alternative bucket.
	size_t i1 = Hash1(fp, mask);
	if (i == i1)
	    i = Hash2(fp, mask);
	else
	    i = i1;
	b = bb + bsize * i;
	// Insert to the alternative bucket.
	if (justAdd1(fp, b, i, bsize))
	    return true;
    } while (maxkick-- > 0);
    // Ran out of tries? ofp already set.
    return false;
}

static inline size_t insertloop(uint64_t *bb, size_t nswap, uint64_t *swap,
	int logsize, size_t mask, int bsize)
{
    size_t nout = 0;
    for (size_t k = 0; k < nswap; k++) {
	uint64_t fp = swap[k];
	dFP2IB(fp, bb, mask);
	if (justAdd2(fp, b1, i1, b2, i2, bsize))
	    continue;
	if (kickAdd(fp, bb, b1, i1, &fp, logsize, mask, bsize))
	    continue;
	swap[nout++] = fp;
    }
    return nout;
}

static inline uint64_t *reinterp23(uint64_t *bb, size_t nb)
{
    // Resizing e.g. 2GB -> 3GB cannot trigger size_t overflow.
    bb = realloc(bb, (3 * nb + SENTINELS) * sizeof(uint64_t));
    if (!bb)
	return NULL;
    memset(A16(bb + 3 * nb), 0xff, SENTINELS * sizeof(uint64_t));

    // Reinterpret as a 3-tier array.
    //
    //             2 3 . .   . . . .
    //   1 2 3 4   1 3 4 .   1 2 3 4
    //   1 2 3 4   1 2 4 .   1 2 3 4

    for (size_t i = nb - 2; i; i -= 2) {
	uint64_t *src0 = bb + 2 * i, *src1 = src0 + 2;
	uint64_t *dst0 = bb + 3 * i, *dst1 = dst0 + 3;
	memcpy(    dst1 , A16(src1), 16); dst1[2] = 0;
	memcpy(A16(dst0), A16(src0), 16); dst0[2] = 0;
    }
    bb[5] = 0, bb[4] = bb[3], bb[3] = bb[2], bb[2] = -1;
    return bb;
}

static inline uint64_t *reinterp34(uint64_t *bb, size_t nb, int logsize)
{
    // On 32-bit platforms, going 3GB -> 4GB will result in size_t overflow.
    if (logsize >= 27 && sizeof(size_t) < 5)
	return errno = ENOMEM, NULL;
    bb = realloc(bb, (4 * nb + SENTINELS) * sizeof(uint64_t));
    if (!bb)
	return NULL;
    memset(A16(bb + 4 * nb), 0xff, SENTINELS * sizeof(uint64_t));

    // Reinterpret as a 4-tier array.
    //
    //             2 3 4 .   . . . .
    //   1 2 3 4   1 3 4 .   1 2 3 4
    //   1 2 3 4   1 2 4 .   1 2 3 4
    //   1 2 3 4   1 2 3 .   1 2 3 4

    for (size_t i = nb - 2; i; i -= 2) {
	uint64_t *src0 = bb + 3 * i, *src1 = src0 + 3;
	uint64_t *dst0 = bb + 4 * i, *dst1 = dst0 + 4;
	dst1[2] = src1[2]; memcpy(A16(dst1),     src1 , 16); dst1[3] = 0;
	dst0[2] = src0[2]; memcpy(A16(dst0), A16(src0), 16); dst0[3] = 0;
    }
    bb[7] = 0, bb[6] = bb[5], bb[5] = bb[4], bb[4] = bb[3], bb[3] = -1;
    return bb;
}

static inline bool t_resize(struct set *set, uint64_t fp, int bsize, bool sse4)
{
    uint64_t *bb = bsize == 2 ?
	    reinterp23(set->bb, set->mask + 1) :
	    reinterp34(set->bb, set->mask + 1, set->logsize);
    if (!bb)
	return false;
    set->bb = bb;

    // Insert fp (no kicks required).
    size_t i = Hash1(fp, set->mask);
    uint64_t *b = bb + (bsize + 1) * i;
    if (b[0] == b[1])
	b[0] = fp;
    else if (b[1] == b[2])
	b[1] = fp;
    else if (bsize == 2 || b[2] == b[3])
	b[2] = fp;
    else
	b[3] = fp;

    // Try to insert the stashed elements.
    set->nstash = insertloop(bb, 2, set->stash, set->logsize, set->mask, bsize + 1);
    // The outcome determines which vtab functions will further be used.
    if (set->nstash == 0) {
	if (bsize == 2)
	    SetVFuncs(set, 3, 0, sse4);
	else
	    SetVFuncs(set, 4, 0, sse4);
	set->cnt += 3;
    }
    else {
	if (bsize == 2)
	    SetVFuncs(set, 3, 1, sse4);
	else
	    SetVFuncs(set, 4, 1, sse4);
	if (set->nstash == 2)
	    set->cnt += 2;
	else {
	    assert(set->nstash == 1);
	    set->stash[1] = set->stash[0];
	    set->cnt++;
	}
    }

    // The data structure upconverted.
    set->bsize = bsize + 1;
    return true;
}

static bool fp64set_resize23(struct set *set, uint64_t fp, bool sse4) { return t_resize(set, fp, 2, sse4); }
static bool fp64set_resize34(struct set *set, uint64_t fp, bool sse4) { return t_resize(set, fp, 3, sse4); }

static inline uint64_t *reinterp43(uint64_t *bb, size_t nb, int logsize)
{
    // The logsize is going up, hitting the hash space limit?
    if (logsize >= 32)
	return errno = E2BIG, NULL;
    bb = realloc(bb, (6 * nb + SENTINELS) * sizeof(uint64_t));
    if (!bb)
	return NULL;
    memset(A16(bb + 6 * nb), 0xff, SENTINELS * sizeof(uint64_t));

    // Reinterpret as a 3-tier array.
    //
    //   x x x x
    //   1 2 3 4   1 2 3 x 4 . . .   1 2 3 4 ? . . .
    //   1 2 3 4   1 2 x 3 4 . . .   1 2 3 4 ? . . .
    //   1 2 3 4   1 x 2 3 4 x . .   1 2 3 4 ? ? . .

    bb[3] = bb[4], bb[4] = bb[5], bb[5] = bb[6];
    bb[6] = bb[8], bb[7] = bb[9], bb[8] = bb[10];
    bb[9] = bb[12], bb[10] = bb[13], bb[11] = bb[14];

    for (size_t i = 4; i < nb; i += 2) {
	uint64_t *src0 = bb + 4 * i, *src1 = src0 + 4;
	uint64_t *dst0 = bb + 3 * i, *dst1 = dst0 + 3;
	memcpy(A16(dst0), A16(src0), 24);
	memcpy(    dst1 , A16(src1), 24);
    }

    // Reassign elements to new slots.
    //
    //   1 2 3 4 . . . .   . . . 4 . . . .
    //   1 2 3 4 . . . .   . 2 . 4 1 . 3 .
    //   1 2 3 4 . . . .   1 2 3 4 1 2 3 .

    size_t mask2 = 2 * nb - 1;
#define HashesTo(fp, j) \
    ((Hash1(fp, mask2) == j) | (Hash2(fp, mask2) == j))

    // When spreading a row, some elements are moved,
    // and some not.  There are eight outcomes.
    //
    //   0 0 0 0 1 1 1 1
    //   0 0 1 1 0 0 1 1
    //   0 1 0 1 0 1 0 1
    //
    //   0 1 2 3 4 5 6 7

#define Spread(i, vblank, wblank)			\
    do {						\
	size_t j = i + nb;				\
	uint64_t *v = bb + 3 * i;			\
	uint64_t *w = bb + 3 * j;			\
	switch (HashesTo(v[0], j) << 0 |		\
		HashesTo(v[1], j) << 1 |		\
		HashesTo(v[2], j) << 2) {		\
	case 0:						\
	    w[0] = w[1] = w[2] = wblank;		\
	    break;					\
	case 1:						\
	    w[0] = v[0], w[1] = w[2] = wblank;		\
	    v[0] = v[1], v[1] = v[2], v[2] = vblank;	\
	    break;					\
	case 2:						\
	    w[0] = v[1], w[1] = w[2] = wblank;		\
	    v[1] = v[2], v[2] = vblank;			\
	    break;					\
	case 3:						\
	    w[0] = v[0], w[1] = v[1], w[2] = wblank;	\
	    v[0] = v[2], v[1] = v[2] = vblank;		\
	    break;					\
	case 4:						\
	    w[0] = v[2], w[1] = w[2] = wblank;		\
	    v[2] = vblank;				\
	    break;					\
	case 5:						\
	    w[0] = v[0], w[1] = v[2], w[2] = wblank;	\
	    v[0] = v[1], v[1] = v[2] = vblank;		\
	    break;					\
	case 6:						\
	    w[0] = v[1], w[1] = v[2], w[2] = wblank;	\
	    v[1] = v[2] = vblank;			\
	    break;					\
	default:					\
	    w[0] = v[0], w[1] = v[1], w[2] = v[2];	\
	    v[0] = v[1] = v[2] = vblank;		\
	}						\
    } while (0)

    Spread(0, -1, 0);
    for (size_t i = 1; i < nb; i++)
	Spread(i, 0, 0);
    return bb;
}

static bool fp64set_resize43(struct set *set, uint64_t fp, bool sse4)
{
    // The only point of deliberate failure:
    // bucket size = 4, fill factor < 50%.
    size_t nb = set->mask + 1;
    if (set->cnt < 2 * nb)
	return errno = EAGAIN, false;

    uint64_t *swap = reallocarray(NULL, nb + 4, sizeof(uint64_t));
    if (!swap)
	return false;
    size_t nswap = 3;
    swap[0] = fp;
    swap[1] = set->stash[0];
    swap[2] = set->stash[1];

    // Swap off the 4th tier, along with fp and the stashed elements.
    //
    //   1 2 3 4   x x x x   swap: fp stash 1 2 3 4
    //   1 2 3 4   1 2 3 4
    //   1 2 3 4   1 2 3 4
    //   1 2 3 4   1 2 3 4

    uint64_t *bb = set->bb;
#define Copy2(i, blank0, blank1)		\
    do {					\
	swap[nswap] = bb[i+3];			\
	nswap      += bb[i+3] != blank0;	\
	swap[nswap] = bb[i+7];			\
	nswap      += bb[i+7] != blank1;	\
    } while (0)

    Copy2(0, -1, 0);
    for (size_t i = 2; i < nb; i += 2)
	Copy2(4*i, 0, 0);

    bb = reinterp43(bb, nb, set->logsize);
    if (!bb) {
	free(swap);
	return false;
    }
    set->bb = bb;

    size_t mask2 = 2 * nb - 1;
    nswap = insertloop(bb, nswap, swap, set->logsize + 1, mask2, 3);

    free(swap);

    assert(nswap == 0);
    assert(set->nstash == 2);
    set->cnt += 3;

    set->mask = mask2;
    set->logsize++;
    set->nstash = 0;
    set->bsize = 3;

    SetVFuncs(set, 3, 0, sse4);

    return true;
}

HIDDEN FP64SET_FASTCALL int fp64set_tail2(uint64_t fp, struct set *set)
{
    assert(set->bsize == 2);
    assert(set->nstash == 2);
    if (fp64set_resize23(set, fp, 1))
	return 2;
    return -1;
}

static inline bool stashAdd(struct set *set, uint64_t fp, bool nstash, int bsize, bool sse4)
{
    // No stash yet?
    if (nstash == 0) {
	set->nstash = 1;
	set->stash[0] = set->stash[1] = fp;
	// Switch vfuncs to the ones with stash.
	SelectVFuncs(set, bsize, 1, sse4);
	return true;
    }
    // Free slot in the stash?
    if (set->nstash == 1) {
	set->nstash = 2;
	set->stash[1] = fp;
	return true;
    }
    // The stash is full.
    return false;
}

// Template for virtual functions.
static inline int t_add(struct set *set, uint64_t fp, bool nstash, int bsize)
{
    dFP2IB(fp, set->bb, set->mask);
    if (has(fp, b1, b2, nstash, set->stash, bsize))
	return 0;
    if (justAdd2(fp, b1, i1, b2, i2, bsize))
	return set->cnt++, 1;
    // A comment on random walk.
    if (kickAdd(fp, set->bb, b1, i1, &fp, set->logsize, set->mask, bsize))
	return set->cnt++, 1;
    bool sse4 = false;
    if (stashAdd(set, fp, nstash, bsize, sse4))
	return 1; // stashing doesn't change set->cnt
    if (nstash && bsize == 2 && fp64set_resize23(set, fp, sse4)) return 2;
    if (nstash && bsize == 3 && fp64set_resize34(set, fp, sse4)) return 2;
    if (nstash && bsize == 4 && fp64set_resize43(set, fp, sse4)) return 2;
    return -1;
}

#if FP64SET_SSE4
static inline SSE4_FUNC int t_add_sse4(struct set *set, uint64_t fp, bool nstash, int bsize)
{
    dFP2IB(fp, set->bb, set->mask);
    if (has_sse4(fp, b1, b2, nstash, set->stash, bsize))
	return 0;
    if (justAdd2(fp, b1, i1, b2, i2, bsize))
	return set->cnt++, 1;
    if (kickAdd(fp, set->bb, b1, i1, &fp, set->logsize, set->mask, bsize))
	return set->cnt++, 1;
    bool sse4 = true;
    if (stashAdd(set, fp, nstash, bsize, sse4))
	return 1;
    if (nstash && bsize == 2 && fp64set_resize23(set, fp, sse4)) return 2;
    if (nstash && bsize == 3 && fp64set_resize34(set, fp, sse4)) return 2;
    if (nstash && bsize == 4 && fp64set_resize43(set, fp, sse4)) return 2;
    return -1;
}
#endif

static inline int t_del(struct set *set, uint64_t fp, bool nstash, int bsize)
{
#define DelBucketRet(b, blank)			\
    return b[bsize-1] = blank, set->cnt--, 1
#define DelBucket(b, blank)			\
    do {					\
	if (b[0] == fp) {			\
	    b[0] = b[1];			\
	    if (bsize > 2) b[1] = b[2];		\
	    if (bsize > 3) b[2] = b[3];		\
	    DelBucketRet(b, blank);		\
	}					\
	if (b[1] == fp) {			\
	    if (bsize > 2) b[1] = b[2];		\
	    if (bsize > 3) b[2] = b[3];		\
	    DelBucketRet(b, blank);		\
	}					\
	if (bsize > 2 && b[2] == fp) {		\
	    if (bsize > 3) b[2] = b[3];		\
	    DelBucketRet(b, blank);		\
	}					\
	if (bsize > 3 && b[3] == fp)		\
	    DelBucketRet(b, blank);		\
    } while (0)

    dFP2IB(fp, set->bb, set->mask);
    uint64_t blank1 = 0 - (i1 == 0);
    uint64_t blank2 = 0 - (i2 == 0);
    DelBucket(b1, blank1);
    DelBucket(b2, blank2);
    if (nstash == 0)
	return 0;
    if (set->stash[0] == fp) {
	// Only one fingerprint in the stash?
	if (set->stash[1] == fp) {
	    set->stash[0] = set->stash[1] = 0;
	    set->nstash = 0;
	    // Switch vfuncs to the ones without stash.
	    SelectVFuncs(set, bsize, 0, cpu_supports_sse4);
	    return 1;
	}
	// Two fingerprints in the stash.
	set->stash[0] = set->stash[1];
	set->stash[1] = 0;
	set->nstash = 1;
	return 1;
    }
    if (set->stash[1] == fp) {
	set->stash[1] = 0;
	set->nstash = 1;
	return 1;
    }
    return 0;
}

static inline uint64_t *t_next(struct set *set, size_t *iter, bool nstash, int bsize)
{
    size_t i = *iter;
    uint64_t *bb = set->bb;
    size_t n = bsize * (set->mask + 1);
    // The first bucket?
    for (; unlikely(i < bsize); i++)
	if (bb[i] != UINT64_MAX)
	    return *iter = i + 1, &bb[i];
    // The rest of the buckets, iterate as flat array.
    while (bb[i] == 0)
	i++;
    if (i < n)
	return *iter = i + 1, &bb[i];
    if (nstash == 0)
	return *iter = 0, NULL;
    if (i == n)
	return *iter = n + 1, &set->stash[0];
    if (i == n + 1 && set->stash[0] != set->stash[1])
	return *iter = n + 2, &set->stash[1];
    return *iter = 0, NULL;
}

FP64SET_FASTCALL const uint64_t *fp64set_next(const struct fp64set *arg, size_t *iter)
{
    struct set *set = (void *) arg;
    return t_next(set, iter, set->nstash, set->bsize);
}

#undef MakeVFuncs
#define MakeVFuncs(NB, ST)				      \
    VFUNC(has##NB##st##ST) { return t_has(set, fp, ST, NB); } \
    VFUNC(add##NB##st##ST) { return t_add(set, fp, ST, NB); } \
    VFUNC(del##NB##st##ST) { return t_del(set, fp, ST, NB); }
MakeAllVFuncs

#if FP64SET_SSE4
#undef MakeVFuncs
#define MakeVFuncs(NB, ST)						 \
    VFUNC(add##NB##st##ST##sse4) { return t_add_sse4(set, fp, ST, NB); }
MakeAllVFuncs
#endif

// ex:set ts=8 sts=4 sw=4 noet:
