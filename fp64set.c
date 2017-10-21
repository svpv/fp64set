#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "fpset.h"

struct fpset {
    // The total number of fingerprints.
    size_t n;
    // The number of buckets, the logarithm.
    unsigned logsize;
    // The buckets; each bucket has 2 fingerprint slots.
    uint64_t (*buckets)[2];
    // Bit vector, indicates which buckets are in use.
    unsigned long *fill;
};

// Implements bit vectors atop of unsigned long words (hence W),
// similar to FD_SET macros in <sys/select.h>.
#define W_BITS (8 * sizeof(unsigned long))
#define W_ELT(n) ((n) / W_BITS)
#define W_MASK(n) (1UL << ((n) % W_BITS))
#define W_SIZE(n) (((n) + (W_BITS - 1)) / W_BITS)
#define W_BYTES(n) (W_SIZE(n) * sizeof(unsigned long))
#define W_SET(w, n) ((void) ((w)[W_ELT(n)] |= W_MASK(n)))
#define W_CLR(w, n) ((void) ((w)[W_ELT(n)] &= ~W_MASK(n)))
#define W_ISSET(w, n) (((w)[W_ELT(n)] & W_MASK(n)) != 0)

// Allocates set->buckets and set->fill in a single chunk.
static bool fpset_alloc(struct fpset *set)
{
    size_t n = (size_t) 1 << set->logsize;
    size_t fillsize = W_BYTES(n);
    size_t allocsize = n * sizeof(*set->buckets) + fillsize;
    set->buckets = malloc(allocsize);
    if (!set->buckets)
	return false;
    set->fill = (void *) (set->buckets + n);
    memset(set->fill, 0, fillsize);
    return true;
}

struct fpset *fpset_new(unsigned logsize)
{
    assert(logsize >= 1);
    // Each bucket takes 16+ bytes.
    assert(logsize < 8 * sizeof(size_t) - 4);
    // Each fingerprint is treated as two 32-bit hashes.
    assert(logsize <= 32);
    struct fpset *set = malloc(sizeof *set);
    if (!set)
	return NULL;
    set->n = 0;
    set->logsize = logsize;
    if (!fpset_alloc(set))
	return free(set), NULL;
    return set;
}

void fpset_free(struct fpset *set)
{
    if (!set)
	return;
    free(set->buckets);
    free(set);
}

// Place a fingerprint at its index; set->n should be updated by the caller.
static int fpset_add1(struct fpset *set, uint64_t fp, size_t i)
{
    // Activate the bucket.
    if (!W_ISSET(set->fill, i)) {
	W_SET(set->fill, i);
	// Set both buckets to the same value, this will indicate
	// that only one bucket out of two is in use.
	set->buckets[i][0] = set->buckets[i][1] = fp;
	return +1;
    }
    // Already added?
    if (fp == set->buckets[i][0] || fp == set->buckets[i][1])
	return 0;
    // Has a free slot?
    if (set->buckets[i][0] == set->buckets[i][1]) {
	set->buckets[i][0] = fp;
	return +1;
    }
    // Both buckets occupied.
    return -1;
}

// As the theory goes, each fingerprint can reside at either of its two buckets.
static int fpset_add2(struct fpset *set, uint64_t fp, size_t i1, size_t i2)
{
    bool isset1 = W_ISSET(set->fill, i1);
    bool isset2 = W_ISSET(set->fill, i2);
    // Already added?
    if (isset1 && (fp == set->buckets[i1][0] || fp == set->buckets[i1][1]))
	return 0;
    if (isset2 && (fp == set->buckets[i2][0] || fp == set->buckets[i2][1]))
	return 0;
    // Activate a bucket.
    if (!isset1) {
	W_SET(set->fill, i1);
	set->buckets[i1][0] = set->buckets[i1][1] = fp;
	return +1;
    }
    if (!isset2) {
	W_SET(set->fill, i2);
	set->buckets[i2][0] = set->buckets[i2][1] = fp;
	return +1;
    }
    // Has a free slot?
    if (set->buckets[i1][0] == set->buckets[i1][1]) {
	set->buckets[i1][0] = fp;
	return +1;
    }
    if (set->buckets[i2][0] == set->buckets[i2][1]) {
	set->buckets[i2][0] = fp;
	return +1;
    }
    return -1;
}

// Add a fingerprint to one of its buckets.  If there's no free slot,
// run the "kick loop" to remap existing fingerprints.  If that fails,
// the existing fingerprint that was kicked out is returned via ofpp.
static int fpset_addk(struct fpset *set, uint64_t fp, uint64_t *ofpp)
{
    size_t mask = ((size_t) 1 << set->logsize) - 1;
    size_t i1 = (fp >> 32) & mask;
    size_t i2 = (i1 ^ fp) & mask;
    int n = fpset_add2(set, fp, i1, i2);
    if (n >= 0) {
	set->n += n;
	return 0;
    }
    // Assume that the fingerprint is the only "entropy" we now have.
    uint64_t seed = fp;
    // Randomly pick i1 or i2.  The fingerprint will be placed in this bucket,
    // and one of the two fingerprints which already occupy the bucket will be
    // "kicked out" to its second bucket.
    size_t i = ((seed >> 37) ^ (seed >> 13)) & 1 ? i1 : i2;
#define MAXKICK 16
    for (int k = 0; k < MAXKICK; k++) {
	// Randomly select bucket[i][0] or bucket[i][1].
	// This is the "other" fingerprint which shall be kicked out.
	size_t j = ((seed >> 33) ^ (seed >> 17)) & 1;
	uint64_t ofp = set->buckets[i][j];
	// Swap fp and ofp.
	set->buckets[i][j] = fp;
	fp = ofp;
	// Try to place the fingerprint to its second bucket.
	i = (i ^ fp) & mask;
	n = fpset_add1(set, fp, i);
	if (n >= 0) {
	    // No dups are possible at this stage.
	    assert(n == 1);
	    set->n += n;
	    return 1;
	}
	// Add the fingerprint to the entropy.
#define rotl64(x, r) ((x << r) | (x >> (64 - r)))
	seed ^= rotl64(fp, 37);
    }
    *ofpp = fp;
    return -1;
}

int fpset_add(struct fpset *set, uint64_t fp)
{
    uint64_t ofp;
    int ret = fpset_addk(set, fp, &ofp);
    if (ret >= 0)
	return 0;
    // If the fill factor is already below 50%, it's a failure.
    if (set->n < (size_t) 1 << set->logsize)
	return -1;
    // Need to rebuild the table with the increased logsize.
    size_t logsize = set->logsize++;
    assert(set->logsize <= 32);
    // Save the existing data.
    uint64_t (*buckets)[2] = set->buckets;
    unsigned long *fill = set->fill;
    // Reallocate the structure.
    if (!fpset_alloc(set)) {
	set->buckets = buckets;
	set->logsize--;
	return -1;
    }
    // Have that many fingerprints, not including the one that was kicked out.
    size_t oldn = set->n;
    // Insert the fingerprint that was kicked out.
    size_t mask = ((size_t) 1 << set->logsize) - 1;
    fpset_add1(set, ofp, (ofp >> 32) & mask);
    set->n = 1;
    // Reinsert the existing fingerprints.
    for (size_t i = 0; i < (size_t) 1 << logsize; i++) {
	if (!W_ISSET(fill, i))
	    continue;
	if (fpset_addk(set, buckets[i][0], &ofp) < 0)
	    return -2;
	if (buckets[i][0] == buckets[i][1])
	    continue;
	if (fpset_addk(set, buckets[i][1], &ofp) < 0)
	    return -2;
    }
    // No dups could have been discovered.
    assert(set->n == oldn + 1);
    // Allocated in a single chunk.
    free(buckets);
    return 1;
}

bool fpset_has(struct fpset *set, uint64_t fp)
{
    size_t mask = ((size_t) 1 << set->logsize) - 1;
    size_t i1 = (fp >> 32) & mask;
    size_t i2 = (i1 ^ fp) & mask;
    if (W_ISSET(set->fill, i1))
	if (fp == set->buckets[i1][0] || fp == set->buckets[i1][1])
	    return true;
    if (W_ISSET(set->fill, i2))
	if (fp == set->buckets[i2][0] || fp == set->buckets[i2][1])
	    return true;
    return false;
}

// ex:set ts=8 sts=4 sw=4 noet:
