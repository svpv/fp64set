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

#ifndef FP64SET_X86_S
#define FP64SET_X86_S

#define AlignFunc  32
#define AlignLoop  32

#define FUNC(name)  \
	.global    NAME(name); \
	.hidden    NAME(name); \
	.type      NAME(name),@function; \
	.align     AlignFunc; \
	NAME(name):

#define END(name) \
	.size      NAME(name),.-NAME(name)

#define m_stash    0
#define m_mask     28
#define m_bb       32
#define m_cnt      36
#define m_logsize  44

#if defined(_WIN32) || defined(__CYGWIN__)
#define MSFASTCALL 1
#else
#define MSFASTCALL 0
#endif

#if MSFASTCALL
#define r_lo       %ecx
#define r_hi       %edx
#define r_ptr      %eax
#define REG3       mov 4(%esp),r_ptr
#define RET        ret $4
#define ALLOC      sub $12,%esp
#define FREE       add $12,%esp
#define SAVE_PTR
#define save_ptr   16
#define save_ebx   0
#define save_esi   4
#define save_edi   8

#else // REGPARM3
#define r_lo       %eax
#define r_hi       %edx
#define r_ptr      %ecx
#define REG3
#define RET        ret
#define ALLOC      sub $16,%esp
#define FREE       add $16,%esp
#define SAVE_PTR   mov r_ptr,(%esp)
#define save_ptr   0
#define save_ebx   4
#define save_esi   8
#define save_edi   12
#endif

// has() only clobbers %XMM0 and %xmm1-%xmm3.
#define XMM0 xmm4

// Common setup for has() and add().
// The fingerprint goes into \xmm0.
.macro argBegin xmm0 st sse4
	movd     r_lo,\xmm0
    .ifnb \sse4
	pinsrd   $1,r_hi,\xmm0
    .else
	movd     r_hi,%xmm1
	punpckldq %xmm1,\xmm0
    .endif
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
    .ifnb \st
	movdqa   m_stash(r_ptr),\st
    .endif
	mov      m_bb(r_ptr),r_ptr
	movdDup  \xmm0,\xmm0
.endm

.macro hasBegin st sse4
	REG3
	argBegin %XMM0,\st,\sse4
.endm

.macro hasEnd xmm
	pmovmskb \xmm,%eax
	RET
.endm

// add() routines start in a manner similar to has().
.macro addBegin st sse4
	REG3
	ALLOC
	SAVE_PTR
	argBegin %xmm0,\st,\sse4
.endm

// If the element is already added, add() returns early.
.macro addEnd xmm insert
	pmovmskb \xmm,r_ptr
	test     r_ptr,r_ptr
	jz       \insert
	FREE
	xor      %eax,%eax
	RET
.endm

// insert() are private routines called by add() with custom calling
// convention: most registers are passed from add() in pristine form;
// specifically b1[] elements are passed in xmm1 and (bsize>2) xmm3.
.macro insertBegin
	// Strategically bump set->cnt.
	mov      save_ptr(%esp),r_ptr
	addl     $1,m_cnt(r_ptr)
	mov      m_bb(r_ptr),r_ptr
	mov      %ebx,save_ebx(%esp)
.endm

// The frist part of insert() is sometimes named justAdd() - the easy case
// when there is a free slot in either b1[] or b2[].  This is how it ends.
.macro justEnd
	mov      save_ebx(%esp),%ebx
	FREE
	mov      $1,%eax
	RET
.endm

// Otherwise, the second part of insert() is entered, the kickAdd() loop.
// An element is kicked out from the bottom of b1[] and reinserted at the
// top of its alternative bucket.
.macro gonnaKick
	mov      %esi,save_esi(%esp)
	mov      %edi,save_edi(%esp)
	mov      save_ptr(%esp),%ebx
	mov      m_mask(%ebx),%esi
	mov      m_logsize(%ebx),%edi
	shl      $1,%edi
.endm

// Successful kickAdd() return.
.macro kickEnd
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	FREE
	mov      $1,%eax
	RET
.endm

// Executed at the end of each (unsuccessful) kickAdd() iteration.
// This further exposes how b1[] insertion works: the element inserted
// at b[r_lo] from the top is in xmm0, and two elements from the bottom
// are in xmm1; xmm1[0] goes to the alternative bucket at b[r_hi];
// xmm1[1] is further used to shift the elements down the bucket.
// During each iteration, b[r_hi] is loaded into xmm2 and (bsize>2) xmm3.
.macro kickMore label tail sse4
	// No free slot at r_hi.
	movq     %xmm1,%xmm0
	movdqa   %xmm2,%xmm1
	mov      r_hi,r_lo
	// Loop control.
	sub      $1,%edi
	jge      \label
// Unsuccessful kickAdd() is followed by the tail call to a C routine,
// which may resize the table.
	mov      save_ptr(%esp),r_ptr
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	FREE
	movd     %xmm0,r_lo
    .ifnb \sse4
	pextrd   $1,%xmm0,r_hi
    .else
	pshufd   $1,%xmm0,%xmm0
	movd     %xmm0,r_hi
    .endif
	jmp      \tail
.endm

.macro kickMore_sse4 label tail
	kickMore \label \tail sse4=1
.endm

#define TAIL(name) fp64set_##name##tail

// In has() routines, we compare quadwords with pcmpeqq (two at a time) and
// combine comparisons with por.  With SSE2 however, we can only compare
// doublewords with pcmpeqd.  We fix the discrepancy by using the modified
// por step: two comparisons are packed in such a way that transforms
// qword-all-1s to dword-all-1s.  It then suffices to check the result
// with pcmpeqd against all-1s.
.macro porQQ xmm2 xmm1
	packssdw \xmm2,\xmm1
	pcmpeqd  \xmm2,\xmm2
	pcmpeqd  \xmm2,\xmm1
.endm

// porQQ can also be viewed as a packing primitive.
.macro packQQ xmm2 xmm1
	porQQ    \xmm2,\xmm1
.endm

// Packing with interleaving.
.macro blendQQ xmm2 xmm1
	packQQ   \xmm2,\xmm1
	// Swap the middle elements.
	pshufd   $0xd8,\xmm1,\xmm1
.endm

// The kicking primitive.  The new item in xmm0 is "pressed" from the top,
// and the resulting xmm0 is supposed to overwrite the memory where xmm1
// comes from; xmm1 is not cobbered.
//       |   |
//       +   +
//  xmm0 | 3 |           +---+
//       +---+           | 3 |
//              ->  xmm0 +   +
//       +---+           | 2 |
//       | 2 |           +---+
//  xmm1 +   +
//       | 1 |
//       +---+
.macro pressQQ xmm1 xmm0
	shufpd   $2,\xmm1,\xmm0
	shufpd   $1,\xmm0,\xmm0
.endm

// Continuing with the above picture, xmm1 corresponds to the bucket
// at bb[r_lo].  The fingerprint "1" has been kiked out of the bucket,
// but xmm1 has not been clobbered.  So we need to find the alternative
// bucket for xmm1[0], to insert from the top; this will be bb[r_hi].
.macro altHi shl sse4
	movd     %xmm1,r_hi
    .ifnb \sse4
	pextrd   $1,%xmm1,%ebx
    .else
	pshufd   $1,%xmm1,%xmm0
	movd     %xmm0,%ebx
    .endif
	and      %esi,r_hi
	and      %esi,%ebx
    .ifnb \shl
	shl      \shl,r_hi
	shl      \shl,%ebx
    .else
	lea      (r_hi,r_hi,2),r_hi
	lea      (%ebx,%ebx,2),%ebx
    .endif
	cmp      r_lo,r_hi
	cmove    %ebx,r_hi
.endm

// A technique similar to porQQ can be used to combine 3 results,
// as if with two por instructions.
.macro porQ3ac xmm2 xmm1
	packssdw \xmm2,\xmm1
.endm
.macro porQ3bc xmm2 xmm1
	packssdw \xmm2,\xmm2
	packsswb \xmm2,\xmm1
	pcmpeqw  \xmm2,\xmm2
	pcmpeqw  \xmm2,\xmm1
.endm

// To combine 4 results, porQQ+porQQ+por can be used, total 3+2+1=6 instructions.
// The following scheme shows, however, that this can be done with 5 instructions.
.macro porQ4ab xmm2 xmm1
	packssdw \xmm2,\xmm1
.endm
.macro porQ4cd xmm2 xmm1
	packssdw \xmm2,\xmm1
.endm
.macro porQ4bd xmm2 xmm1
	packsswb \xmm2,\xmm1
	pcmpeqw  \xmm2,\xmm2
	pcmpeqw  \xmm2,\xmm1
.endm

// Finally, any comparison can be combined with previously packed results.
// This relies on the fact that packssdw preserves non-zero in the low part;
// the high part is permuted and ANDed: [1a,1b,2a,2b] x [1b,1a,2b,2a].
.macro porQD xmm2 xmm1
	packssdw \xmm2,\xmm1
	pshufhw  $0xb1,\xmm1,\xmm2
	pand     \xmm2,\xmm1
.endm

// Unoccupied slots have blank values.  The blank value for the bucket at bb[0]
// is UINT64_MAX, the blank value for all other buckets is 0.
.macro setBlank lohi xmm setAll0s useAll0s
	movd     \lohi,\xmm
    .ifnb \setAll0s
	pxor     \setAll0s,\setAll0s
	pcmpeqd  \setAll0s,\xmm
    .else
	pcmpeqd  \useAll0s,\xmm
    .endif
	pshufd   $0,\xmm,\xmm
.endm

.macro setBlank_sse4 lohi xmm setAll0s useAll0s
	movd     \lohi,\xmm
	movddup  \xmm,\xmm
    .ifnb \setAll0s
	pxor     \setAll0s,\setAll0s
	pcmpeqq  \setAll0s,\xmm
    .else
	pcmpeqq  \useAll0s,\xmm
    .endif
.endm

// SSE2 routines
#define NAME(name) fp64set_##name##sse2
#define movdDup punpcklqdq
#define pcmpeqQ pcmpeqd

#include "fp64set-x86.S"

#undef NAME
#undef movDdup
#undef pcmpeqQ

// SSE4 routines
#define SSE4 1
#define NAME(name) fp64set_##name##sse4
#define porQQ   por
#define porQD   por
#define porQ3ac por
#define porQ3bc por
#define porQ4ab por
#define porQ4cd por
#define porQ4bd por
#define movSD   pblendw $0x0f,
#define packQQ  packssdw
#define blendQQ pblendw $0xcc,
#define pressQQ palignr $8,
#define altHi   altHi sse4=1
#define hasBegin hasBegin sse4=1
#define andBegin andBegin sse4=1
#define setBlank setBlank_sse4
#define kickMore kickMore_sse4

#endif // FP64SET_X86_S

FUNC(has2st0)
	hasBegin
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm1
	pcmpeqQ  %XMM0,%xmm1
	pcmpeqQ  (r_ptr,r_hi,1),%XMM0
	porQQ    %XMM0,%xmm1
	hasEnd   %xmm1
END(has2st0)

FUNC(add2st0)
	addBegin
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm3
	movdqa   (r_ptr,r_hi,1),%xmm4
	movdqa   %xmm3,%xmm1
	pcmpeqQ  %xmm0,%xmm3
	movdqa   %xmm4,%xmm2
	pcmpeqQ  %xmm0,%xmm4
	porQQ    %xmm4,%xmm3
	addEnd   %xmm3 NAME(insert2)
END(add2st0)

FUNC(insert2)
	insertBegin

	// Check for free slots in b1[*] -> xmm3.
	setBlank r_lo,%xmm3 setAll0s=%xmm5
	pcmpeqQ  %xmm1,%xmm3

	// Check for free slots in b2[*] -> xmm4.
	setBlank r_hi,%xmm4 useAll0s=%xmm5
	pcmpeqQ  %xmm2,%xmm4

	// Combine (b1[0],b2[0],b1[1],b2[1]) -> mask.
	blendQQ  %xmm4,%xmm3
	pmovmskb %xmm3,%ebx
	test     %ebx,%ebx
	jnz      1f

	gonnaKick
	jmp      2f

1:	bsf      %ebx,%ebx
	// Is it b1[] or b2[]? bsf returns 0, 4, 8, or 12.
	test     $4,%ebx
	cmovnz   r_hi,r_lo
	// Is it [0] or [1]?
	shr      $3,%ebx
	lea      (r_lo,%ebx,8),r_lo
	movq     %xmm0,(r_ptr,r_lo,1)
	justEnd

	.align   AlignLoop
2:	pressQQ  %xmm1,%xmm0
	movdqa   %xmm0,(r_ptr,r_lo,1)

	altHi    shl=$4

	// Check for empty slots at bb[r_hi].
	setBlank r_hi,%xmm3 useAll0s=%xmm5
	movdqa   (r_ptr,r_hi,1),%xmm2
	pcmpeqQ  %xmm2,%xmm3
    #if !SSE4
	pshufd   $0xb1,%xmm3,%xmm0
	pand     %xmm0,%xmm3
    #endif
	pmovmskb %xmm3,%ebx
	test     %ebx,%ebx
	jnz      3f

	kickMore 2b TAIL(insert2)

3:	bsf      %ebx,%ebx
	add      %ebx,r_hi
	movq     %xmm1,(r_ptr,r_hi,1)
	kickEnd
END(insert2)

FUNC(add2st1)
	addBegin st=%xmm5
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm3
	movdqa   (r_ptr,r_hi,1),%xmm4
	pcmpeqQ  %xmm0,%xmm5
	movdqa   %xmm3,%xmm1
	pcmpeqQ  %xmm0,%xmm3
	porQ3ac  %xmm5,%xmm3
	movdqa   %xmm4,%xmm2
	pcmpeqQ  %xmm0,%xmm4
	porQ3bc  %xmm4,%xmm3
	addEnd   %xmm3 NAME(insert2)
END(add2st1)

FUNC(has2st1)
	hasBegin st=%xmm3
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm1
	pcmpeqQ  %XMM0,%xmm3
	pcmpeqQ  %XMM0,%xmm1
	pcmpeqQ  (r_ptr,r_hi,1),%XMM0
	porQ3ac  %xmm3,%xmm1
	porQ3bc  %XMM0,%xmm1
	hasEnd   %xmm1
END(has2st1)

FUNC(has3st0)
	hasBegin
	lea      (r_lo,r_lo,2),r_lo
	lea      (r_hi,r_hi,2),r_hi
	movdqu   8(r_ptr,r_lo,8),%xmm1
	movdqu   8(r_ptr,r_hi,8),%xmm2
	pcmpeqQ  %XMM0,%xmm1
	movq     (r_ptr,r_lo,8),%xmm3
	movhps   (r_ptr,r_hi,8),%xmm3
	pcmpeqQ  %XMM0,%xmm2
	porQ3ac  %xmm2,%xmm1
	pcmpeqQ  %XMM0,%xmm3
	porQ3bc  %xmm3,%xmm1
	hasEnd   %xmm1
END(has3st0)

FUNC(add3st0)
	addBegin
	lea      (r_lo,r_lo,2),r_lo
	lea      (r_hi,r_hi,2),r_hi
	movdqu   (r_ptr,r_lo,8),%xmm4
	movdqu   (r_ptr,r_hi,8),%xmm5
	movq     16(r_ptr,r_lo,8),%xmm3
	movhps   16(r_ptr,r_hi,8),%xmm3
	movdqa   %xmm4,%xmm1
	pcmpeqQ  %xmm0,%xmm4
	movdqa   %xmm5,%xmm2
	pcmpeqQ  %xmm0,%xmm5
	porQ3ac  %xmm5,%xmm4
	movdqa   %xmm3,%xmm5
	pcmpeqQ  %xmm0,%xmm5
	porQ3bc  %xmm5,%xmm4
	addEnd   %xmm4 NAME(insert3)
END(add3st0)

FUNC(insert3)
	insertBegin

	// Numbers correspond to XMM registers.
	// +---+---+
	// |   3   |
	// +---+---+
	// |   |   |
	// + 1 + 2 +
	// |   |   |
	// +---+---+
	//   ^   ^
	//  buckets

	// Check for free slots in b2[0,1] -> xmm2.
	setBlank r_hi,%xmm6 setAll0s=%xmm5
	pcmpeqQ  %xmm6,%xmm2

	// Prepare blank values for b1[*] -> xmm4, b1[*] + b2[*] -> xmm6.
	setBlank r_lo,%xmm4 useAll0s=%xmm5
	movSD    %xmm4,%xmm6

	// Check for free slots in b1[0,1] -> xmm4, b1[2] + b2[2] -> xmm6.
	pcmpeqQ  %xmm1,%xmm4
	pcmpeqQ  %xmm3,%xmm6

	// Combine (b1[0],b2[0],b1[1],b2[1],b1[2],b2[2],x,x) -> mask.
    #if SSE4
	pshufd   $0x88,%xmm6,%xmm6
	pblendw  $0xcc,%xmm2,%xmm4
	packssdw %xmm6,%xmm4
    #else
	packssdw %xmm6,%xmm6
	packssdw %xmm2,%xmm4
	pshufd   $0xd8,%xmm4,%xmm4
	packsswb %xmm6,%xmm4
	pcmpeqw  %xmm2,%xmm2
	pcmpeqw  %xmm2,%xmm4
    #endif
	pmovmskb %xmm4,%ebx
	test     %ebx,%ebx
	jnz      1f

	gonnaKick
	jmp      2f

1:	bsf      %ebx,%ebx
	// Is it b1[] or b2[]? bsf returns (0, 4, 8) vs (2, 6, 10).
	test     $2,%ebx
	cmovnz   r_hi,r_lo
	// Is it [0], [1] or [2]?
	shr      $2,%ebx
	add      %ebx,r_lo
	movq     %xmm0,(r_ptr,r_lo,8)
	justEnd

	.align   AlignLoop
2:	pressQQ  %xmm1,%xmm3
	movdqu   %xmm3,(r_ptr,r_lo,8)
	movq     %xmm0,16(r_ptr,r_lo,8)

	altHi

	// Check for empty slots at bb[r_hi].
	setBlank r_hi,%xmm4 useAll0s=%xmm5
	movdqu   (r_ptr,r_hi,8),%xmm2
	movq     16(r_ptr,r_hi,8),%xmm3
	movdqa   %xmm2,%xmm0
	pcmpeqQ  %xmm4,%xmm0
	pcmpeqQ  %xmm3,%xmm4
	packQQ   %xmm4,%xmm0
	pmovmskb %xmm0,%ebx
	and      $0x0fff,%ebx
	jnz      3f

	kickMore 2b TAIL(insert3)

3:	bsf      %ebx,%ebx
	shr      $2,%ebx
	add      %ebx,r_hi
	movq     %xmm1,(r_ptr,r_hi,8)
	kickEnd
END(insert3)

FUNC(add3st1)
	addBegin st=%xmm5
	lea      (r_lo,r_lo,2),r_lo
	lea      (r_hi,r_hi,2),r_hi
	movdqu   (r_ptr,r_lo,8),%xmm4
	pcmpeqQ  %xmm0,%xmm5
	movdqa   %xmm4,%xmm1
	pcmpeqQ  %xmm0,%xmm4
	porQ4ab  %xmm5,%xmm4
	movdqu   (r_ptr,r_hi,8),%xmm5
	movq     16(r_ptr,r_lo,8),%xmm6
	movhps   16(r_ptr,r_hi,8),%xmm6
	movdqa   %xmm5,%xmm2
	pcmpeqQ  %xmm0,%xmm5
	movdqa   %xmm6,%xmm3
	pcmpeqQ  %xmm0,%xmm6
	porQ4cd  %xmm6,%xmm5
	porQ4bd  %xmm5,%xmm4
	addEnd   %xmm4 NAME(insert3)
END(add3st1)

FUNC(has3st1)
	hasBegin st=%xmm3
	lea      (r_lo,r_lo,2),r_lo
	lea      (r_hi,r_hi,2),r_hi
	movdqu   8(r_ptr,r_lo,8),%xmm1
	movdqu   8(r_ptr,r_hi,8),%xmm2
	pcmpeqQ  %XMM0,%xmm3
	pcmpeqQ  %XMM0,%xmm1
	porQ4ab  %xmm3,%xmm1
	movq     (r_ptr,r_lo,8),%xmm3
	movhps   (r_ptr,r_hi,8),%xmm3
	pcmpeqQ  %XMM0,%xmm2
	pcmpeqQ  %XMM0,%xmm3
	porQ4cd  %xmm3,%xmm2
	porQ4bd  %xmm2,%xmm1
	hasEnd   %xmm1
END(has3st1)

FUNC(has4st0)
	hasBegin
	shl      $5,r_lo
	shl      $5,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm1
	movdqa   (r_ptr,r_hi,1),%xmm2
	pcmpeqQ  %XMM0,%xmm1
	movdqa   16(r_ptr,r_lo,1),%xmm3
	pcmpeqQ  %XMM0,%xmm2
	pcmpeqQ  %XMM0,%xmm3
	pcmpeqQ  16(r_ptr,r_hi,1),%XMM0
	porQ4ab  %xmm2,%xmm1
	porQ4cd  %xmm3,%XMM0
	porQ4bd  %XMM0,%xmm1
	hasEnd   %xmm1
END(has4st0)

FUNC(add4st0)
	addBegin
	shl      $5,r_lo
	shl      $5,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm5
	movdqa   (r_ptr,r_hi,1),%xmm6
	movdqa   %xmm5,%xmm1
	pcmpeqQ  %xmm0,%xmm5
	movdqa   %xmm6,%xmm2
	pcmpeqQ  %xmm0,%xmm6
	porQ4ab  %xmm6,%xmm5
	movdqa   16(r_ptr,r_lo,1),%xmm6
	movdqa   16(r_ptr,r_hi,1),%xmm7
	movdqa   %xmm6,%xmm3
	pcmpeqQ  %xmm0,%xmm6
	movdqa   %xmm7,%xmm4
	pcmpeqQ  %xmm0,%xmm7
	porQ4cd  %xmm7,%xmm6
	porQ4bd  %xmm6,%xmm5
	addEnd   %xmm5 NAME(insert4)
END(add4st0)

FUNC(insert4)
	insertBegin

	// +---+---+
	// |   |   |
	// + 3 + 4 +
	// |   |   |
	// +---+---+
	// |   |   |
	// + 1 + 2 +
	// |   |   |
	// +---+---+

	// Check for free slots in b2[*] -> xmm2.
	setBlank r_hi,%xmm6 setAll0s=%xmm5
	pcmpeqQ  %xmm6,%xmm2
	pcmpeqQ  %xmm6,%xmm4
	packssdw %xmm4,%xmm2

	// Check for free slots in b1[*] -> xmm6.
	setBlank r_lo,%xmm6 useAll0s=%xmm5
	movdqa   %xmm3,%xmm4
	pcmpeqQ  %xmm6,%xmm4
	pcmpeqQ  %xmm1,%xmm6
	packssdw %xmm4,%xmm6

	// Combine (b1[0],b2[0],b1[1],b2[1],b1[2],b2[2],b1[3],b2[3]) -> mask.
    #if SSE4
	pblendw  $0xaa,%xmm2,%xmm6
    #else
	// SSE2 can't blend, so self-pack and interleve the low halves.
	packsswb %xmm6,%xmm6
	packsswb %xmm2,%xmm2
	punpcklwd %xmm2,%xmm6
	// Compensate for missing pcmpeqQ.
	pcmpeqw  %xmm2,%xmm2
	pcmpeqw  %xmm2,%xmm6
    #endif
	pmovmskb %xmm6,%ebx
	test     %ebx,%ebx
	jnz      1f

	gonnaKick
	jmp      2f

1:	bsf      %ebx,%ebx
	// Is it b1[] or b2[]? bsf returns (0, 4, 8, 12) vs (2, 6, 10, 14).
	test     $2,%ebx
	cmovnz   r_hi,r_lo
	// Is it [0], [1], [2], or [3]?
	shr      $2,%ebx
	lea      (r_lo,%ebx,8),r_lo
	movq     %xmm0,(r_ptr,r_lo,1)
	justEnd

	.align   AlignLoop
2:	pressQQ  %xmm3,%xmm0
	movdqa   %xmm0,16(r_ptr,r_lo,1)
	pressQQ  %xmm1,%xmm3
	movdqa   %xmm3,(r_ptr,r_lo,1)

	altHi    shl=$5

	// Check for empty slots at bb[r_hi].
	setBlank r_hi,%xmm4 useAll0s=%xmm5
	movdqa   (r_ptr,r_hi,1),%xmm2
	movdqa   16(r_ptr,r_hi,1),%xmm3
	movdqa   %xmm2,%xmm0
	pcmpeqQ  %xmm4,%xmm0
	pcmpeqQ  %xmm3,%xmm4
	packQQ   %xmm4,%xmm0
	pmovmskb %xmm0,%ebx
	test     %ebx,%ebx
	jnz      3f

	kickMore 2b TAIL(insert4)

3:	bsf      %ebx,%ebx
	lea      (r_hi,%ebx,2),r_hi
	movq     %xmm1,(r_ptr,r_hi,1)
	kickEnd
END(insert4)

FUNC(add4st1)
	addBegin st=%xmm7
	shl      $5,r_lo
	shl      $5,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm5
	movdqa   (r_ptr,r_hi,1),%xmm6
	pcmpeqQ  %xmm0,%xmm7
	movdqa   %xmm5,%xmm1
	pcmpeqQ  %xmm0,%xmm5
	porQ4ab  %xmm7,%xmm5
	movdqa   16(r_ptr,r_lo,1),%xmm7
	movdqa   %xmm6,%xmm2
	pcmpeqQ  %xmm0,%xmm6
	movdqa   %xmm7,%xmm3
	pcmpeqQ  %xmm0,%xmm7
	porQ4cd  %xmm7,%xmm6
	movdqa   16(r_ptr,r_hi,1),%xmm7
	porQ4bd  %xmm6,%xmm5
	movdqa   %xmm7,%xmm4
	pcmpeqQ  %xmm0,%xmm7
	porQD    %xmm7,%xmm5
	addEnd   %xmm5 NAME(insert4)
END(add4st1)

FUNC(has4st1)
	hasBegin st=%xmm3
	shl      $5,r_lo
	shl      $5,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm1
	movdqa   (r_ptr,r_hi,1),%xmm2
	pcmpeqQ  %XMM0,%xmm3
	pcmpeqQ  %XMM0,%xmm1
	porQ4ab  %xmm3,%xmm1
	movdqa   16(r_ptr,r_lo,1),%xmm3
	pcmpeqQ  %XMM0,%xmm2
	pcmpeqQ  %XMM0,%xmm3
	pcmpeqQ  16(r_ptr,r_hi,1),%XMM0
	porQ4cd  %xmm3,%xmm2
	porQ4bd  %xmm2,%xmm1
	porQD    %XMM0,%xmm1
	hasEnd   %xmm1
END(has4st1)
