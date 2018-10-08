#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <assert.h>
#include <x86intrin.h>
#include "fp64set.h"

static inline uint64_t rotr64(uint64_t x, int r)
{
    return (x >> r) | (x << (64 - r));
}

static uint64_t rndState = 16294208416658607535ULL;

static inline uint64_t rnd(void)
{
    uint64_t ret = rotr64(rndState, 16);
    rndState = rndState * 6364136223846793005ULL + 1442695040888963407ULL;
    return ret;
}

// Add random elements in a loop, until the structure resizes.
void addUniq(struct fp64set *set, size_t *np, uint64_t *tp)
{
    size_t n = 0;
    uint64_t state0 = rndState;
    uint64_t rnd1, rnd2;
    uint64_t t0 = __rdtsc();
    uint64_t t1 = t0;
    while (1) {
	rnd1 = rnd();
	int rc = fp64set_add(set, rnd1);
	assert(rc > 0);
	// Not including the resize cost.
	if (rc > 1)
	    break;
	n++;
	t1 = __rdtsc();
    }
    // Recheck with fp64set_has().
    rndState = state0;
    do {
	rnd2 = rnd();
	assert(fp64set_has(set, rnd2));
    } while (rnd2 != rnd1);
    *np = n;
    *tp = t1 - t0;
}

// From MurmurHash3.
static inline uint64_t fmix64(uint64_t x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

// To produce dups, we simply mask out the high bits.  But for the structure
// to work, all the bits have to look random.  This is where the diffusion
// function, fmix64, comes to rescue.  All its operations are reversible,
// so we've got a bijection from the set of small numbers to the uniformly
// distributed subset of the 64-bit universe.
void addDups(struct fp64set *set, uint64_t mask, size_t *np, uint64_t *tp)
{
    size_t n = 0;
    uint64_t t0 = __rdtsc();
    uint64_t t1 = t0;
    while (1) {
	uint64_t fp = fmix64(rnd() & mask);
	int rc = fp64set_add(set, fp);
	assert(rc >= 0);
	if (rc > 1)
	    break;
	n++;
	t1 = __rdtsc();
    }
    *np = n;
    *tp = t1 - t0;
}

static int ITER = 23;

double bench_addUniq(int bsize, int logsize)
{
    size_t n = 0; uint64_t t = 0;
    for (int i = 0; i < (1<<ITER); i++) {
	struct fp64set *set = fp64set_new(logsize);
	size_t n1 = 0; uint64_t t1 = 0;
	// Skip the stages preceding bsize.
	for (int i = 2; i <= bsize; i++)
	    addUniq(set, &n1, &t1);
	n += n1, t += t1;
	fp64set_free(set);
    }
    return (double) t / n;
}

double bench_addDups(int bsize, int logsize)
{
    size_t n = 0; uint64_t t = 0;
    for (int i = 0; i < (1<<ITER); i++) {
	struct fp64set *set = fp64set_new(logsize);
	size_t n1 = 0; uint64_t t1 = 0;
	// If the structure has 2^b slots, use b-bit random numbers.
	// This will produce as many dups as possible without looping
	// indefinitely (because the fill factor falls short of 100%).
	uint64_t mask = (1 << (logsize + bsize - 1)) - 1;
	for (int i = 2; i <= bsize; i++)
	    addDups(set, mask, &n1, &t1);
	n += n1, t += t1;
	fp64set_free(set);
    }
    return (double) t / n;
}

double bench_has(int bsize, int logsize)
{
    size_t n = 0; uint64_t t = 0;
    struct fp64set *set = fp64set_new(logsize);
    // has() is branchless, so in fact the contents do not matter.
    // Only add some to switch to the right bsize.
    for (int i = 2; i < bsize; i++)
	addUniq(set, &n, &t);
    n = 1 << (logsize + ITER);
    t = __rdtsc();
    size_t dummy = 0;
    for (size_t i = 0; i < n; i++)
	dummy += fp64set_has(set, rnd());
    t = __rdtsc() - t;
    return (double) (t + dummy % 2) / n;
}

double bench_del(int bsize, int logsize)
{
    size_t n = 0; uint64_t t = 0;
    for (int i = 0; i < (1<<ITER); i++) {
	struct fp64set *set = fp64set_new(logsize);
	size_t fill = 0;
	uint64_t saveRndState = rndState;
	// Switch to the right bsize.
	for (int i = 2; i < bsize; i++) {
	    size_t nx = 0; uint64_t tx = 0;
	    addUniq(set, &nx, &tx);
	    // The element that switched the structure didn't count.
	    fill += nx + 1;
	}
	// Need 50% fill factor.
	size_t need = (1 << logsize) * bsize / 2;
	while (fill < need) {
	    int rc = fp64set_add(set, rnd());
	    assert(rc == 1);
	    fill += 1;
	}
	rndState = saveRndState;
	uint64_t tx = __rdtsc();
	for (size_t i = 0; i < fill; i++) {
	    bool deleted = fp64set_del(set, rnd());
	    assert(deleted);
	}
	tx = __rdtsc() - tx;
	n += fill, t += tx;
	// Try to delete non-existent, should fail.
	bool deleted = fp64set_del(set, rnd());
	assert(!deleted);
	fp64set_free(set);
    }
    return (double) t / n;
}

double bench_next(int bsize, int logsize)
{
    size_t n = 0; uint64_t t = 0;
    for (int i = 0; i < (1<<ITER); i++) {
	struct fp64set *set = fp64set_new(logsize);
	size_t fill = 0;
	for (int i = 2; i < bsize; i++) {
	    size_t nx = 0; uint64_t tx = 0;
	    addUniq(set, &nx, &tx);
	    fill += nx + 1;
	}
	size_t need = (1 << logsize) * bsize / 2;
	while (fill < need) {
	    int rc = fp64set_add(set, rnd());
	    assert(rc == 1);
	    fill += 1;
	}
	uint64_t tx = __rdtsc();
	size_t iter = 0;
	size_t niter = 0;
	while (fp64set_next(set, &iter))
	    niter++;
	tx = __rdtsc() - tx;
	assert(niter == fill);
	n += fill, t += tx;
	fp64set_free(set);
    }
    return (double) t / n;
}

int main(int argc, char **argv)
{
    int nb = 10;
    if (argc > 1) {
	const char *arg1 = argv[1];
	unsigned char c = *arg1;
	if (c >= '0' && c <= '9') {
	    nb = atoi(arg1);
	    assert(nb >= 3);
	    assert(nb <= 16); // good rnd bits
	    argc--, argv++;
	}
    }
    ITER -= nb;
    bool ALL = argc <= 1;
    ITER += !ALL;
    bool b_has2 = ALL, b_has3 = ALL, b_has4 = ALL;
    bool b_add2u = ALL, b_add3u = ALL, b_add4u = ALL;
    bool b_add2d = ALL, b_add3d = ALL, b_add4d = ALL;
    bool b_del2 = ALL, b_del3 = ALL, b_del4 = ALL;
    bool b_next2 = ALL, b_next3 = ALL, b_next4 = ALL;
    for (int i = 1; !ALL && i < argc; i++) {
	if (0) continue;
	else if (strcmp(argv[i], "has") == 0) b_has2 = b_has3 = b_has4 = 1;
	else if (strcmp(argv[i], "addu") == 0) b_add2u = b_add3u = b_add4u = 1;
	else if (strcmp(argv[i], "addd") == 0) b_add2d = b_add3d = b_add4d = 1;
	else if (strcmp(argv[i], "del") == 0) b_del2 = b_del3 = b_del4 = 1;
	else if (strcmp(argv[i], "next") == 0) b_next2 = b_next3 = b_next4 = 1;
	else if (strcmp(argv[i], "has2") == 0) b_has2 = 1;
	else if (strcmp(argv[i], "has3") == 0) b_has3 = 1;
	else if (strcmp(argv[i], "has4") == 0) b_has4 = 1;
	else if (strcmp(argv[i], "add2u") == 0) b_add2u = 1;
	else if (strcmp(argv[i], "add3u") == 0) b_add3u = 1;
	else if (strcmp(argv[i], "add4u") == 0) b_add4u = 1;
	else if (strcmp(argv[i], "add2d") == 0) b_add2d = 1;
	else if (strcmp(argv[i], "add3d") == 0) b_add3d = 1;
	else if (strcmp(argv[i], "add4d") == 0) b_add4d = 1;
	else if (strcmp(argv[i], "del2") == 0) b_del2 = 1;
	else if (strcmp(argv[i], "del3") == 0) b_del3 = 1;
	else if (strcmp(argv[i], "del4") == 0) b_del4 = 1;
	else if (strcmp(argv[i], "next2") == 0) b_next2 = 1;
	else if (strcmp(argv[i], "next3") == 0) b_next3 = 1;
	else if (strcmp(argv[i], "next4") == 0) b_next4 = 1;
    }
    if (b_add2u) printf("add2 uniq %.2f\n", bench_addUniq(2, nb));
    if (b_add3u) printf("add3 uniq %.2f\n", bench_addUniq(3, nb));
    if (b_add4u) printf("add4 uniq %.2f\n", bench_addUniq(4, nb));
    // NB: dups incur extra costs of fmix64.
    if (b_add2d) printf("add2 dups %.2f\n", bench_addDups(2, nb));
    if (b_add3d) printf("add3 dups %.2f\n", bench_addDups(3, nb));
    if (b_add4d) printf("add4 dups %.2f\n", bench_addDups(4, nb));
    if (b_has2) printf("has2 %.2f\n", bench_has(2, nb));
    if (b_has3) printf("has3 %.2f\n", bench_has(3, nb));
    if (b_has4) printf("has4 %.2f\n", bench_has(4, nb));
    if (b_del2) printf("del2 %.2f\n", bench_del(2, nb));
    if (b_del3) printf("del3 %.2f\n", bench_del(3, nb));
    if (b_del4) printf("del4 %.2f\n", bench_del(4, nb));
    // NB: price drops as the fill factor increases.
    if (b_next2) printf("next2 %.2f\n", bench_next(2, nb));
    if (b_next3) printf("next3 %.2f\n", bench_next(3, nb));
    if (b_next4) printf("next4 %.2f\n", bench_next(4, nb));
    return 0;
}
