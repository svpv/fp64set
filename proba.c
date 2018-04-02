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

// The PRNG state is initialized with the first outputs of splitmix64
// seeded with zero (much like suggested in xoroshiro128plus.c).
static uint64_t prngState = 0xe220a8397b1dcdaf;

#include <string.h>
#include <sys/auxv.h>

static void randomizePrng(void)
{
    void *auxrnd = (void *) getauxval(AT_RANDOM);
    assert(auxrnd);
    memcpy(&prngState, auxrnd, sizeof prngState);
}

// A fast LCG-based PRNG, medium quality.  Just happens to be good enough
// for the job.  Another reason for using this generator is that its state
// is only a single 64-bit variable, and so it yields unique 64-bit numbers.
// This makes it possible to disable dup detection during simulations.
static inline uint64_t rnd(void)
{
    // Use the previous state as the basis for the return value, to improve
    // instruction-level parallelism.  Rotate the worst 12 bits away into
    // the higher half.  Would like to rotate even more, possibly by 16 bits,
    // but need to keep at least 20 good bits in the higher half to address
    // up to 1M buckets.
    uint64_t ret = rotr64(prngState, 12);
    // The state is updated in parallel even as the caller makes use of "ret"
    // to index into the buckets.  Constants are Knuth's.
    prngState = prngState * 0x5851f42d4c957f2d + 0x14057b7ef767814f;
    return ret;
}

// A simplified fpset_add() clone, returns the number of kicks.
static inline int proba_add(struct fpset *set, uint64_t fp)
{
    dFP2IB;
    // No dups during simulations, because of full-period LCG.
    if (justAdd(fp, b1, i1, b2, i2))
	return set->cnt++, 0;
    return kickAdd(set, fp, b1, i1, &fp);
}

#include <stdio.h>
#include "qsort.h"

// Estimate the fill factor achievable 99% of the time.
static void fillfactor(void)
{
    int logfix = FPSET_BUCKETSIZE > 2;
    for (int logsize = 4 + logfix; logsize <= 16 + logfix; logsize++) {
	int tries[4000];
	for (int i = 0; i < 4000; i++) {
	    struct fpset *set = fpset_new(logsize);
	    int added = 0;
	    while (proba_add(set, rnd()) >= 0)
		added++;
	    tries[i] = added;
	    fpset_free(set);
	}
	int try1;
#define tryLess(i, j) tries[i] < tries[j]
#define trySwap(i, j) try1 = tries[i], tries[i] = tries[j], tries[j] = try1
	QSORT(4000, tryLess, trySwap);
	size_t i = sizeof(tries)/sizeof(int)/100; // 100 for q=1%, 4 for q=25%
	double q = (tries[i-1] + tries[i-2] + tries[i-3] + tries[i-4] +
		    tries[i+0] + tries[i+1] + tries[i+2] + tries[i+3]) / 8.0;
	int bucketlog = logsize - logfix;
	int slots = (1 << bucketlog) * FPSET_BUCKETSIZE;
	printf("bucketlog=%d\tslots=%d\tq=%.1f\tfillfactor=%.3f\n",
		bucketlog, slots, q, q / slots);

    }
}

#include <getopt.h>

enum {
    OPT_RANDOMIZE = 256,
};

static const struct option longopts[] = {
    { "randomize", no_argument, NULL, OPT_RANDOMIZE },
};

int main(int argc, char **argv)
{
    int c;
    while ((c = getopt_long(argc, argv, "h", longopts, NULL)) != -1)
	switch (c) {
	case 0:
	    break;
	case OPT_RANDOMIZE:
	    randomizePrng();
	    break;
	default:
	    fprintf(stderr, "Usage: %s [OPTIONS...] [ARGS...]\n", argv[0]);
	    return 1;
	}
    fillfactor();
    return 0;
}
