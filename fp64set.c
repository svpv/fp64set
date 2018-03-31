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

#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <errno.h>
#include "fpset.h"

// Slots per bucket (2, 3, or 4).
#ifndef FPSET_BUCKETSIZE
#define FPSET_BUCKETSIZE 2
#elif FPSET_BUCKETSIZE < 2 || FPSET_BUCKETSIZE > 4
#error "bad FPSET_BUCKETSIZE value"
#endif

struct fpset {
    // The total number of unique fingerprints added.
    size_t cnt;
    // The number of buckets, the logarithm.
    int logsize;
    // The number of buckets - 1, helps indexing into the buckets.
    size_t mask;
    // The buckets (malloc'd); each bucket has a few fingerprint slots.
    uint64_t (*bb)[FPSET_BUCKETSIZE];
};

// Make two indexes out of a fingerprint.
// Fingerprints are treated as two 32-bit hash values for this purpose.
#define FP2I(fp, mask)		\
    i1 = (fp >> 00) & mask;	\
    i2 = (fp >> 32) & mask
// Further identify the buckets.
#define FP2IB			\
    FP2I(fp, set->mask);	\
    b1 = set->bb[i1];		\
    b2 = set->bb[i2]
// Further declare vars.
#define dFP2IB			\
    size_t i1, i2;		\
    uint64_t *b1, *b2;		\
    FP2IB

#define unlikely(cond) __builtin_expect(cond, 0)

// Check if a fingerprint has already been inserted.  I trust gcc to unroll
// the loop properly.  Note that only two memory locations are accessed
// (which translates into only two cache lines, unless FPSET_BUCKETSIZE is 3);
// this is one reason why fpset_has() is 2-3 times faster than
// std::unordered_set<uint64_t>::find().
static inline bool has(uint64_t fp, uint64_t *b1, uint64_t *b2)
{
#ifdef FPSET_PROBA
    // This works faster when running Monte Carlo simulations to estimate
    // the probability of failure (in this case, fingerprints are different
    // and equality almost never holds).
    for (int j = 0; j < FPSET_BUCKETSIZE; j++)
	if (unlikely(fp == b1[j] || fp == b2[j]))
	    return true;
    return false;
#else
    // This works faster with real data, where equality sometimes holds.
    // However, if the CPU tries to bet on when and where equality holds,
    // it loses.  It is best to circumvent branch prediction entirely.
    bool has = false;
    for (int j = 0; j < FPSET_BUCKETSIZE; j++)
	has |= (fp == b1[j]) | (fp == b2[j]);
    return has;
#endif
}

// Test if a fingerprint at bb[i][*] is actually a free slot.
// Note that a bucket can only keep hold of such a fingerprint that hashes
// into the bucket.  This obviates the need for separate bookkeeping.
static inline bool freeSlot(uint64_t fp, size_t i)
{
    // Slots must be initialized to 0, except that
    // bb[0][*] slots must be initialized to -1.
    return fp == 0 - (i == 0);
}

// Ensure that logsize (the number of buckets) is not too big.
bool cklogsize(int logsize)
{
    if (sizeof(size_t) > 4) {
	// The only limitation on 64-bit platforms is that we can run out
	// of 32-bit hash values.  With FPSET_BUCKETSIZE=2, the buckets would
	// occupy 16 bytes * 4G = 64G RAM, which currently (as of late 2017)
	// matches maximum RAM supported by high-end desktop CPUs.
	// Furthermore, it doesn't make much sense to build even bigger tables
	// of 64-bit fingerprints, because, exactly at this point, birthday
	// collisions start to take off (and so fingerprints become less useful
	// for identifying data items).
	return logsize <= 32;
    }
    // The limit is more stringent on 32-bit systems.  We only check that the
    // buckets occupy less than 2^32 bytes (or, in other words, that the byte
    // count would fit into size_t); malloc may further impose its own limits.
    // If a bucket occupies 16 bytes, this reduces the 32-bit space by 4 bits;
    // the check then goes like "logsize < 28" which translates into
    // "your appetites be must less than 4G which is the entire address space".
    // Thus logsize=27 limits the bucket memory by 2G for FPSET_BUCKETSIZE=2
    // and 3G for FPSET_BUCKETSIZE=3; FPSET_BUCKETSIZE=4 needs a correction.
    return logsize < CHAR_BIT * sizeof(size_t) - 4 - (FPSET_BUCKETSIZE > 3);
}

struct fpset *fpset_new(int logsize)
{
    assert(logsize >= 0);
    // The number of fingerprints -> the number of buckets.
    logsize -= FPSET_BUCKETSIZE > 2;
    // TODO: tune the minimum number of buckets for some probability.
    if (logsize < 4)
	logsize = 4;
    else if (!cklogsize(logsize))
	return errno = E2BIG, NULL;
    struct fpset *set = malloc(sizeof *set);
    if (!set)
	return NULL;
    set->cnt = 0;
    set->logsize = logsize;
    set->mask = (size_t) 1 << logsize;
    set->bb = calloc(set->mask--, sizeof *set->bb);
    if (!set->bb)
	return free(set), NULL;
    for (int n = 0; n < FPSET_BUCKETSIZE; n++)
	set->bb[0][n] = -1;
    return set;
}

void fpset_free(struct fpset *set)
{
    if (!set)
	return;
#ifdef DEBUG
    // The number of fingerprints must match the occupied slots.
    size_t cnt = 0;
    for (size_t i = 0; i <= set->mask; i++)
	for (int n = 0; n < FPSET_BUCKETSIZE; n++)
	    cnt += !freeSlot(set->bb[i][n], i);
    assert(set->cnt == cnt);
#endif
    free(set->bb);
    free(set);
}

bool fpset_has(struct fpset *set, uint64_t fp)
{
    dFP2IB;
    return has(fp, b1, b2);
}

// That much one needs to know upon the first reading.
// The reset is fpset_add() stuff.

// When trying to add to a non-last slot in a bucket,
// there is even a simpler way to check if that slot is occupied.
static inline bool addNonLast1(uint64_t fp, uint64_t *b, size_t n)
{
    if (b[n] == b[n+1])
	return b[n] = fp, true;
    return false;
}

// When adding to the last slot in a bucket, need to use freeSlot.
static inline bool addLast1(uint64_t fp, uint64_t *b, size_t i, size_t n)
{
    if (freeSlot(b[n], i))
	return b[n] = fp, true;
    return false;
}

#define addNonLast2(fp, b1, b2, n) \
	addNonLast1(fp, b1, n) || \
	addNonLast1(fp, b2, n)
#define addLast2(fp, b1, i1, b2, i2, n) \
	addLast1(fp, b1, i1, n) || \
	addLast1(fp, b2, i2, n)

// Try to add a fingerprint to either of its buckets.
static inline bool justAdd(uint64_t fp, uint64_t *b1, uint64_t *b2)
{
    for (int n = 0; n < FPSET_BUCKETSIZE-1; n++)
	if (addNonLast2(fp, b1, b2, n))
	    return true;
    return false;
}

// I found it's better to refactor justAdd() as a macro, otherwise
// the code works slower (possibly because of too many arguments).
#define justAdd(fp, b1, i1, b2, i2) \
	justAdd(fp, b1, b2) || \
	addLast2(fp, b1, i1, b2, i2, FPSET_BUCKETSIZE-1)

// When all slots for a fingerprint are occupied, insertion "kicks out"
// an already existing fingerprint and tries to place it into an alternative
// slot, this triggering a series of kicks.  On success, returns the number
// of kicks taken.  Returns -1 on with the kicked-out fingerprint in ofp.
static inline int kickAdd(struct fpset *set, uint64_t fp, uint64_t *b, size_t i, uint64_t *ofp)
{
#ifdef FPSET_MAXKICK
    int maxk = FPSET_MAXKICK;
#else
    int maxk = 2 * set->logsize;
#endif
    for (int k = 1; k <= maxk; k++) {
	// Put at the top, kick out from the bottom.
	// Using *ofp as a temporary register.
	*ofp = b[0];
	for (int i = 0; i < FPSET_BUCKETSIZE-1; i++)
	    b[i] = b[i+1];
	b[FPSET_BUCKETSIZE-1] = fp, fp = *ofp;
	// Ponder over the fingerprint that's been kicked out.
	dFP2IB;
	// Find out the alternative bucket.
	if (i == i1)
	    i = i2, b = b2;
	else
	    i = i1, b = b1;
	for (int n = 0; n < FPSET_BUCKETSIZE-1; n++)
	    if (addNonLast1(fp, b, n))
		return set->cnt++, k;
	if (addLast1(fp, b, i, FPSET_BUCKETSIZE-1))
	    return set->cnt++, k;
    }
    // Ran out of tries? ofp already set.
    return -1;
}

int fpset_add(struct fpset *set, uint64_t fp)
{
    dFP2IB;
    if (has(fp, b1, b2))
	return 0;
    if (justAdd(fp, b1, i1, b2, i2))
	return set->cnt++, 1;
    if (kickAdd(set, fp, b1, i1, &fp) >= 0)
	return 1;
    return -1;
    // TODO: rebuild the table with logsize + 1.
}

// ex:set ts=8 sts=4 sw=4 noet:
