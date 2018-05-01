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
#include <errno.h>
#include "fp64set.h"

// The real fp64set structure (upconverted from "void *set").
struct set {
    // Virtual functions.
    int (*add)(void *set, uint64_t fp);
    bool (*has)(void *set, uint64_t fp);
    // The number of buckets - 1, helps indexing into the buckets.
    size_t mask;
    // The buckets (malloc'd); each bucket has bsize slots.
    // Two-dimensional structure is emulated with pointer arithmetic.
    uint64_t *bb;
    // To reduce the failure rate, one or two fingerprints can be stashed.
    // When only one fingerprint is stashed, we have stash[0] == stash[1].
    uint64_t stash[2];
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
static inline bool has(uint64_t fp, uint64_t *b1, uint64_t *b2,
	bool nstash, uint64_t *stash, int bsize)
{
    // Issue loads for both buckets.
    bool has1 = fp == b1[0];
    bool has2 = fp == b2[0];
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

// Template for set->has virtual functions.
static inline bool t_has(struct set *set, uint64_t fp, bool nstash, int bsize)
{
    dFP2IB(fp, set->bb, set->mask);
    return has(fp, b1, b2, nstash, set->stash, bsize);
}

// Virtual functions for set->has, differ by the number of slots in a bucket
// and by whether the stash is active.
static bool has2st0(void *set, uint64_t fp) { return t_has(set, fp, 0, 2); }
static bool has2st1(void *set, uint64_t fp) { return t_has(set, fp, 1, 2); }
static bool has3st0(void *set, uint64_t fp) { return t_has(set, fp, 0, 3); }
static bool has3st1(void *set, uint64_t fp) { return t_has(set, fp, 1, 3); }
static bool has4st0(void *set, uint64_t fp) { return t_has(set, fp, 0, 4); }
static bool has4st1(void *set, uint64_t fp) { return t_has(set, fp, 1, 4); }

// Virtual functions for set->add, forward declaration.
static int add2st0(void *set, uint64_t fp);
static int add2st1(void *set, uint64_t fp);
static int add3st0(void *set, uint64_t fp);
static int add3st1(void *set, uint64_t fp);
static int add4st0(void *set, uint64_t fp);
static int add4st1(void *set, uint64_t fp);

struct fp64set *fp64set_new(int logsize)
{
    assert(logsize >= 0);
    if (logsize < 4)
	logsize = 4;
    else if (logsize > 32)
	return errno = E2BIG, NULL;
    else if (logsize > 31 && sizeof(size_t) < 5)
	return errno = ENOMEM, NULL;

    // Starting with two slots per bucket.
    size_t nb = (size_t) 1 << logsize;
    uint64_t *bb = calloc(nb, 2 * sizeof(uint64_t));
    if (!bb)
	return NULL;

    // The blank value for bb[0][*] slots is UINT64_MAX.
    bb[0] = bb[1] = UINT64_MAX;

    struct set *set = malloc(sizeof *set);
    if (!set)
	return free(bb), NULL;

    set->add = add2st0;
    set->has = has2st0;
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

void fp64set_free(struct fp64set *arg)
{
    struct set *set = (void *) arg;
    if (!set)
	return;
#ifdef DEBUG
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
    free(set->bb);
    free(set);
}

// That much one needs to know upon the first reading.
// The reset is fp64set_add() stuff.

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
    int maxk = 2 * set->logsize;
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
