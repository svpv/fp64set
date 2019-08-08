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

#if defined(__i386__) || defined(__ILP32__)
#define m_stash    0
#define m_mask     28
#define m_bb       32
#define m_cnt      36
#define m_logsize  44
#else
#define m_stash    0
#define m_mask     40
#define m_bb       48
#define m_cnt      56
#define m_logsize  68
#endif

#if defined(__i386__)
#if defined(_WIN32) || defined(__CYGWIN__)
#define MSFASTCALL 1
#define REGPARM3   0
#else
#define MSFASTCALL 0
#define REGPARM3   1
#endif
#define MS64ABI    0
#else // x86_64
#if defined(_WIN32) || defined(__CYGWIN__)
#define MS64ABI    1
#else
#define MS64ABI    0
#endif
#define MSFASTCALL 0
#define REGPARM3   0
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
#endif

#if REGPARM3
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

#ifdef __i386__
#define e_lo       r_lo
#define e_hi       r_hi
#define r_bb       r_ptr
#define r_tmp      %ebx
#define e_tmp      %ebx
#define e_mask     %esi
#define e_loop     %edi

#elif MS64ABI
#define q_fp       %rcx
#define r_ptr      %rdx
#define r_lo       %rcx
#define q_lo       %rcx
#define e_lo       %ecx
#define r_hi       %r11
#define q_hi       %r11
#define e_hi       %r11d
#define r_bb       %rax
#define r_tmp      %r8
#define q_tmp      %r8
#define e_tmp      %r8d
#define e_mask     %r9d
#define q_loop     %r10
#define e_loop     %r10d

#else // System V ABI

#ifdef __ILP32__
#define r_ptr      %esi
#define r_lo       %edi
#define r_hi       %r11d
#define r_bb       %eax
#define r_tmp      %edx
#else
#define r_ptr      %rsi
#define r_lo       %rdi
#define r_hi       %r11
#define r_bb       %rax
#define r_tmp      %rdx
#endif

#define q_fp       %rdi
#define q_lo       %rdi
#define e_lo       %edi
#define q_hi       %r11
#define e_hi       %r11d
#define q_tmp      %rdx
#define e_tmp      %edx
#define e_mask     %ecx
#define q_loop     %r8
#define e_loop     %r8d

#endif // System V ABI

#if MS64ABI
// xmm6 and xmm7 are callee-saved, the caller provides the shadow space.
#define saveXmm6 movdqa %xmm6,0x08(%rsp)
#define saveXmm7 movdqa %xmm7,0x18(%rsp)
#define restoreXmm6 movdqa 0x08(%rsp),%xmm6
#define restoreXmm7 movdqa 0x18(%rsp),%xmm7
#else
#define saveXmm6
#define saveXmm7
#define restoreXmm6
#define restoreXmm7
#endif

// Common setup for has() and add().
// The fingerprint goes into \xmm0.
.macro argBegin xmm0 st nomask
    #ifdef __i386__
	movd     e_lo,\xmm0
	pinsrd   $1,e_hi,\xmm0
	and      m_mask(r_ptr),e_lo
	and      m_mask(r_ptr),e_hi
    #else
	mov      q_lo,q_hi
	movq     q_fp,\xmm0
	shr      $32,q_hi
    .ifnb \nomask
	and      m_mask(r_ptr),e_lo
	and      m_mask(r_ptr),e_hi
    .else
	mov      m_mask(r_ptr),e_mask
	and      e_mask,e_lo
	and      e_mask,e_hi
    .endif
    #endif
    .ifnb \st
	movdqa   m_stash(r_ptr),\st
    .endif
	mov      m_bb(r_ptr),r_bb
	movddup  \xmm0,\xmm0
.endm

.macro hasBegin st
    #ifdef __i386__
	REG3
    #endif
	argBegin %xmm0,\st nomask=1
.endm

.macro hasEnd xmm
	pmovmskb \xmm,%eax
	RET
.endm

// add() routines start in a manner similar to has().
.macro addBegin st
    #ifdef __i386__
	REG3
	ALLOC
	SAVE_PTR
    #endif
	argBegin %xmm0,\st
.endm

// If the element is already added, add() returns early.
.macro addEnd xmm insert usingXmm6
    #ifdef __i386__
	pmovmskb \xmm,r_ptr
	test     r_ptr,r_ptr
	jz       \insert
	FREE
    #else
	pmovmskb \xmm,e_loop
	test     e_loop,e_loop
	jz       \insert
    #endif
    .ifnb \usingXmm6
	restoreXmm6
    .endif
	xor      %eax,%eax
	RET
.endm

// insert() are private routines called by add() with custom calling
// convention: most registers are passed from add() in pristine form;
// specifically b1[] elements are passed in xmm1 and (bsize>2) xmm3.
.macro insertBegin
    #if defined(__i386__)
	// The pointer has been clobbered in addEnd.
	mov      save_ptr(%esp),r_ptr
	// Strategically bump set->cnt.
	addl     $1,m_cnt(r_ptr)
	mov      m_bb(r_ptr),r_bb
	mov      %ebx,save_ebx(%esp)
    #else
	addq     $1,m_cnt(r_ptr)
    #endif
.endm

// The frist part of insert() is sometimes named justAdd() - the easy case
// when there is a free slot in either b1[] or b2[].  This is how it ends.
.macro justEnd
    #ifdef __i386__
	mov      save_ebx(%esp),%ebx
	FREE
    #endif
	mov      $1,%eax
	RET
.endm

// Otherwise, the second part of insert() is entered, the kickAdd() loop.
// An element is kicked out from the bottom of b1[] and reinserted at the
// top of its alternative bucket.
.macro gonnaKick
    #ifdef __i386__
	mov      %esi,save_esi(%esp)
	mov      %edi,save_edi(%esp)
	mov      save_ptr(%esp),%ebx
	mov      m_mask(%ebx),e_mask
	mov      m_logsize(%ebx),e_loop
    #else
	// e_mask already loaded in argBegin.
	mov      m_logsize(r_ptr),e_loop
    #endif
	shl      $1,e_loop
.endm

// Successful kickAdd() return.
.macro kickEnd
    #ifdef __i386__
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	FREE
    #endif
	mov      $1,%eax
	RET
.endm

// Executed at the end of each (unsuccessful) kickAdd() iteration.
// This further exposes how b1[] insertion works: the element inserted
// at b[r_lo] from the top is in xmm0, and two elements from the bottom
// are in xmm1; xmm1[0] goes to the alternative bucket at b[r_hi];
// xmm1[1] is further used to shift the elements down the bucket.
// During each iteration, b[r_hi] is loaded into xmm2 and (bsize>2) xmm3.
.macro kickMore label tail
	// No free slot at r_hi.
	movq     %xmm1,%xmm0
	movdqa   %xmm2,%xmm1
	mov      r_hi,r_lo
	// Loop control.
	sub      $1,e_loop
	jge      \label
// Unsuccessful kickAdd() is followed by the tail call to a C routine,
// which may resize the table.
    #ifdef __i386__
    #if REGPARM3
	mov      save_ptr(%esp),r_ptr
    #endif
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	FREE
	movd     %xmm0,e_lo
	pextrd   $1,%xmm0,e_hi
    #else
	movq     %xmm0,q_fp
    #endif
	jmp      \tail
.endm

#define TAIL(name) fp64set_##name##tail

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
	palignr  $8,\xmm1,\xmm0
.endm

.macro preAltHi scale
	mov      (r_bb,r_lo,\scale),e_hi
	mov      4(r_bb,r_lo,\scale),e_tmp
.endm

// Continuing with the above picture, xmm1 corresponds to the bucket
// at bb[r_lo].  The fingerprint "1" has been kiked out of the bucket,
// but xmm1 has not been clobbered.  So we need to find the alternative
// bucket for xmm1[0], to insert from the top; this will be bb[r_hi].
.macro altHi shl
	and      e_mask,e_hi
	and      e_mask,e_tmp
    .ifnb \shl
	shl      \shl,r_hi
	shl      \shl,r_tmp
    .else
	lea      (r_hi,r_hi,2),r_hi
	lea      (r_tmp,r_tmp,2),r_tmp
    .endif
	cmp      r_lo,r_hi
	cmove    r_tmp,r_hi
.endm

// Unoccupied slots have blank values.  The blank value for the bucket at bb[0]
// is UINT64_MAX, the blank value for all other buckets is 0.
.macro setBlank lohi blank
	cmp      $1,\lohi
	sbb      \blank,\blank
.endm

.macro xmmBlank lohi xmm setAll0s useAll0s
    #if defined(__i386__) || defined(__ILP32__)
	movd     \lohi,\xmm
    #else
	movq     \lohi,\xmm
    #endif
	movddup  \xmm,\xmm
    .ifnb \setAll0s
	pxor     \setAll0s,\setAll0s
	pcmpeqq  \setAll0s,\xmm
    .else
	pcmpeqq  \useAll0s,\xmm
    .endif
.endm

#define NAME(name) fp64set_##name##sse4

FUNC(has2st0)
	hasBegin
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_bb,r_lo,1),%xmm1
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  (r_bb,r_hi,1),%xmm0
	por      %xmm0,%xmm1
	hasEnd   %xmm1
END(has2st0)

FUNC(add2st0)
	addBegin
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_bb,r_lo,1),%xmm3
	movdqa   (r_bb,r_hi,1),%xmm4
	movdqa   %xmm3,%xmm1
	pcmpeqq  %xmm0,%xmm3
	movdqa   %xmm4,%xmm2
	pcmpeqq  %xmm0,%xmm4
	por      %xmm4,%xmm3
	addEnd   %xmm3 NAME(insert2)
END(add2st0)

FUNC(insert2)
	insertBegin

	// Check for free slots in b1[*] -> xmm3.
	xmmBlank r_lo,%xmm3 setAll0s=%xmm5
	pcmpeqq  %xmm1,%xmm3

	// Check for free slots in b2[*] -> xmm4.
	xmmBlank r_hi,%xmm4 useAll0s=%xmm5
	pcmpeqq  %xmm2,%xmm4

	// Combine (b1[0],b2[0],b1[1],b2[1]) -> mask.
	pblendw  $0xcc,%xmm4,%xmm3
	pmovmskb %xmm3,e_tmp
	test     e_tmp,e_tmp
	jnz      1f

	gonnaKick
	jmp      NAME(kick2)

1:	bsf      e_tmp,e_tmp
	// Is it b1[] or b2[]? bsf returns 0, 4, 8, or 12.
	test     $4,e_tmp
	cmovnz   r_hi,r_lo
	// Is it [0] or [1]?
	shr      $3,e_tmp
	lea      (r_lo,r_tmp,8),r_lo
	movq     %xmm0,(r_bb,r_lo,1)
	justEnd
END(insert2)

FUNC(kick2)
	pressQQ  %xmm1,%xmm0
	preAltHi scale=1
	movdqa   %xmm0,(r_bb,r_lo,1)

	altHi    shl=$4

#ifdef __i386__
	xmmBlank r_hi,%xmm3 useAll0s=%xmm5
	movdqa   (r_bb,r_hi,1),%xmm2
	pcmpeqq  %xmm2,%xmm3
	pmovmskb %xmm3,e_tmp
	test     e_tmp,e_tmp
	jnz      1f

	kickMore NAME(kick2) TAIL(insert2)

1:	bsf      e_tmp,e_tmp
	add      r_tmp,r_hi
	movq     %xmm1,(r_bb,r_hi,1)
	kickEnd
#else
	setBlank r_hi,q_tmp
	cmp      (r_bb,r_hi,1),q_tmp
	je       1f
	cmp      8(r_bb,r_hi,1),q_tmp
	je       2f

	movdqa   (r_bb,r_hi,1),%xmm2
	kickMore NAME(kick2) TAIL(insert2)

1:	movq     %xmm1,(r_bb,r_hi,1)
	kickEnd
2:	movq     %xmm1,8(r_bb,r_hi,1)
	kickEnd
#endif
END(kick2)

FUNC(add2st1)
	addBegin st=%xmm5
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_bb,r_lo,1),%xmm3
	movdqa   (r_bb,r_hi,1),%xmm4
	pcmpeqq  %xmm0,%xmm5
	movdqa   %xmm3,%xmm1
	pcmpeqq  %xmm0,%xmm3
	por      %xmm5,%xmm3
	movdqa   %xmm4,%xmm2
	pcmpeqq  %xmm0,%xmm4
	por      %xmm4,%xmm3
	addEnd   %xmm3 NAME(insert2)
END(add2st1)

FUNC(has2st1)
	hasBegin st=%xmm3
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_bb,r_lo,1),%xmm1
	pcmpeqq  %xmm0,%xmm3
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  (r_bb,r_hi,1),%xmm0
	por      %xmm3,%xmm1
	por      %xmm0,%xmm1
	hasEnd   %xmm1
END(has2st1)

FUNC(has3st0)
	hasBegin
	lea      (r_lo,r_lo,2),r_lo
	lea      (r_hi,r_hi,2),r_hi
	movdqu   8(r_bb,r_lo,8),%xmm1
	movdqu   8(r_bb,r_hi,8),%xmm2
	pcmpeqq  %xmm0,%xmm1
	movq     (r_bb,r_lo,8),%xmm3
	movhps   (r_bb,r_hi,8),%xmm3
	pcmpeqq  %xmm0,%xmm2
	por      %xmm2,%xmm1
	pcmpeqq  %xmm0,%xmm3
	por      %xmm3,%xmm1
	hasEnd   %xmm1
END(has3st0)

FUNC(add3st0)
	addBegin
	lea      (r_lo,r_lo,2),r_lo
	lea      (r_hi,r_hi,2),r_hi
	movdqu   (r_bb,r_lo,8),%xmm4
	movdqu   (r_bb,r_hi,8),%xmm5
	movq     16(r_bb,r_lo,8),%xmm3
	movhps   16(r_bb,r_hi,8),%xmm3
	movdqa   %xmm4,%xmm1
	pcmpeqq  %xmm0,%xmm4
	movdqa   %xmm5,%xmm2
	pcmpeqq  %xmm0,%xmm5
	por      %xmm5,%xmm4
	movdqa   %xmm3,%xmm5
	pcmpeqq  %xmm0,%xmm5
	por      %xmm5,%xmm4
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
	saveXmm6
	xmmBlank r_hi,%xmm6 setAll0s=%xmm5
	pcmpeqq  %xmm6,%xmm2

	// Prepare blank values for b1[*] -> xmm4, b1[*] + b2[*] -> xmm6.
	xmmBlank r_lo,%xmm4 useAll0s=%xmm5
	pblendw  $0x0f,%xmm4,%xmm6

	// Check for free slots in b1[0,1] -> xmm4, b1[2] + b2[2] -> xmm6.
	pcmpeqq  %xmm1,%xmm4
	pcmpeqq  %xmm3,%xmm6

	// Combine (b1[0],b2[0],b1[1],b2[1],b1[2],b2[2],x,x) -> mask.
	pshufd   $0x88,%xmm6,%xmm6
	pblendw  $0xcc,%xmm2,%xmm4
	packssdw %xmm6,%xmm4
	pmovmskb %xmm4,e_tmp
	restoreXmm6
	test     e_tmp,e_tmp
	jnz      1f

	gonnaKick
	jmp      NAME(kick3)

1:	bsf      e_tmp,e_tmp
	// Is it b1[] or b2[]? bsf returns (0, 4, 8) vs (2, 6, 10).
	test     $2,e_tmp
	cmovnz   r_hi,r_lo
	// Is it [0], [1] or [2]?
	shr      $2,e_tmp
	add      r_tmp,r_lo
	movq     %xmm0,(r_bb,r_lo,8)
	justEnd
END(insert3)

FUNC(kick3)
	pressQQ  %xmm1,%xmm3
	preAltHi scale=8
	movdqu   %xmm3,(r_bb,r_lo,8)
	movq     %xmm0,16(r_bb,r_lo,8)

	altHi

#ifdef __i386__
	xmmBlank r_hi,%xmm4 useAll0s=%xmm5
	movdqu   (r_bb,r_hi,8),%xmm2
	movq     16(r_bb,r_hi,8),%xmm3
	movdqa   %xmm2,%xmm0
	pcmpeqq  %xmm4,%xmm0
	pcmpeqq  %xmm3,%xmm4
	packssdw %xmm4,%xmm0
	pmovmskb %xmm0,e_tmp
	and      $0x0fff,e_tmp
	jnz      1f

	kickMore NAME(kick3) TAIL(insert3)

1:	bsf      e_tmp,e_tmp
	shr      $2,e_tmp
	add      r_tmp,r_hi
	movq     %xmm1,(r_bb,r_hi,8)
	kickEnd
#else
	setBlank r_hi,q_tmp
	cmp      (r_bb,r_hi,8),q_tmp
	je       1f
	cmp      8(r_bb,r_hi,8),q_tmp
	je       2f
	cmp      16(r_bb,r_hi,8),q_tmp
	je       3f

	movdqu   (r_bb,r_hi,8),%xmm2
	movq     16(r_bb,r_hi,8),%xmm3
	kickMore NAME(kick3) TAIL(insert3)

1:	movq     %xmm1,(r_bb,r_hi,8)
	kickEnd
2:	movq     %xmm1,8(r_bb,r_hi,8)
	kickEnd
3:	movq     %xmm1,16(r_bb,r_hi,8)
	kickEnd
#endif
END(kick3)

FUNC(add3st1)
	saveXmm6
	addBegin st=%xmm5
	lea      (r_lo,r_lo,2),r_lo
	lea      (r_hi,r_hi,2),r_hi
	movdqu   (r_bb,r_lo,8),%xmm4
	pcmpeqq  %xmm0,%xmm5
	movdqa   %xmm4,%xmm1
	pcmpeqq  %xmm0,%xmm4
	por      %xmm5,%xmm4
	movdqu   (r_bb,r_hi,8),%xmm5
	movq     16(r_bb,r_lo,8),%xmm6
	movhps   16(r_bb,r_hi,8),%xmm6
	movdqa   %xmm5,%xmm2
	pcmpeqq  %xmm0,%xmm5
	movdqa   %xmm6,%xmm3
	pcmpeqq  %xmm0,%xmm6
	por      %xmm6,%xmm5
	por      %xmm5,%xmm4
	restoreXmm6
	addEnd   %xmm4 NAME(insert3)
END(add3st1)

FUNC(has3st1)
	hasBegin st=%xmm3
	lea      (r_lo,r_lo,2),r_lo
	lea      (r_hi,r_hi,2),r_hi
	movdqu   8(r_bb,r_lo,8),%xmm1
	movdqu   8(r_bb,r_hi,8),%xmm2
	pcmpeqq  %xmm0,%xmm3
	pcmpeqq  %xmm0,%xmm1
	por      %xmm3,%xmm1
	movq     (r_bb,r_lo,8),%xmm3
	movhps   (r_bb,r_hi,8),%xmm3
	pcmpeqq  %xmm0,%xmm2
	pcmpeqq  %xmm0,%xmm3
	por      %xmm3,%xmm2
	por      %xmm2,%xmm1
	hasEnd   %xmm1
END(has3st1)

FUNC(has4st0)
	hasBegin
	shl      $5,r_lo
	shl      $5,r_hi
	movdqa   (r_bb,r_lo,1),%xmm1
	movdqa   (r_bb,r_hi,1),%xmm2
	pcmpeqq  %xmm0,%xmm1
	movdqa   16(r_bb,r_lo,1),%xmm3
	pcmpeqq  %xmm0,%xmm2
	pcmpeqq  %xmm0,%xmm3
	pcmpeqq  16(r_bb,r_hi,1),%xmm0
	por      %xmm2,%xmm1
	por      %xmm3,%xmm0
	por      %xmm0,%xmm1
	hasEnd   %xmm1
END(has4st0)

FUNC(add4st0)
	addBegin
	shl      $5,r_lo
	shl      $5,r_hi
	saveXmm6
	saveXmm7
	movdqa   (r_bb,r_lo,1),%xmm5
	movdqa   (r_bb,r_hi,1),%xmm6
	movdqa   %xmm5,%xmm1
	pcmpeqq  %xmm0,%xmm5
	movdqa   %xmm6,%xmm2
	pcmpeqq  %xmm0,%xmm6
	por      %xmm6,%xmm5
	movdqa   16(r_bb,r_lo,1),%xmm6
	movdqa   16(r_bb,r_hi,1),%xmm7
	movdqa   %xmm6,%xmm3
	pcmpeqq  %xmm0,%xmm6
	movdqa   %xmm7,%xmm4
	pcmpeqq  %xmm0,%xmm7
	por      %xmm7,%xmm6
	por      %xmm6,%xmm5
	restoreXmm7
	addEnd   %xmm5 NAME(insert4) usingXmm6=1
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
	xmmBlank r_hi,%xmm6 setAll0s=%xmm5
	pcmpeqq  %xmm6,%xmm2
	pcmpeqq  %xmm6,%xmm4
	packssdw %xmm4,%xmm2

	// Check for free slots in b1[*] -> xmm6.
	xmmBlank r_lo,%xmm6 useAll0s=%xmm5
	movdqa   %xmm3,%xmm4
	pcmpeqq  %xmm6,%xmm4
	pcmpeqq  %xmm1,%xmm6
	packssdw %xmm4,%xmm6

	// Combine (b1[0],b2[0],b1[1],b2[1],b1[2],b2[2],b1[3],b2[3]) -> mask.
	pblendw  $0xaa,%xmm2,%xmm6
	pmovmskb %xmm6,e_tmp
	restoreXmm6
	test     e_tmp,e_tmp
	jnz      1f

	gonnaKick
	jmp      NAME(kick4)

1:	bsf      e_tmp,e_tmp
	// Is it b1[] or b2[]? bsf returns (0, 4, 8, 12) vs (2, 6, 10, 14).
	test     $2,e_tmp
	cmovnz   r_hi,r_lo
	// Is it [0], [1], [2], or [3]?
	shr      $2,e_tmp
	lea      (r_lo,r_tmp,8),r_lo
	movq     %xmm0,(r_bb,r_lo,1)
	justEnd
END(insert4)

FUNC(kick4)
	pressQQ  %xmm3,%xmm0
	preAltHi scale=1
	movdqa   %xmm0,16(r_bb,r_lo,1)
	pressQQ  %xmm1,%xmm3
	movdqa   %xmm3,(r_bb,r_lo,1)

	altHi    shl=$5

#ifdef __i386__
	xmmBlank r_hi,%xmm4 useAll0s=%xmm5
	movdqa   (r_bb,r_hi,1),%xmm2
	movdqa   16(r_bb,r_hi,1),%xmm3
	movdqa   %xmm2,%xmm0
	pcmpeqq  %xmm4,%xmm0
	pcmpeqq  %xmm3,%xmm4
	packssdw %xmm4,%xmm0
	pmovmskb %xmm0,e_tmp
	test     e_tmp,e_tmp
	jnz      1f

	kickMore NAME(kick4) TAIL(insert4)

1:	bsf      e_tmp,e_tmp
	lea      (r_hi,r_tmp,2),r_hi
	movq     %xmm1,(r_bb,r_hi,1)
	kickEnd
#else
	setBlank r_hi,q_tmp
	cmp      (r_bb,r_hi,1),q_tmp
	je       1f
	cmp      8(r_bb,r_hi,1),q_tmp
	je       2f
	cmp      16(r_bb,r_hi,1),q_tmp
	je       3f
	cmp      24(r_bb,r_hi,1),q_tmp
	je       4f

	movdqa   (r_bb,r_hi,1),%xmm2
	movdqa   16(r_bb,r_hi,1),%xmm3
	kickMore NAME(kick4) TAIL(insert4)

1:	movq     %xmm1,(r_bb,r_hi,1)
	kickEnd
2:	movq     %xmm1,8(r_bb,r_hi,1)
	kickEnd
3:	movq     %xmm1,16(r_bb,r_hi,1)
	kickEnd
4:	movq     %xmm1,24(r_bb,r_hi,1)
	kickEnd
#endif
END(kick4)

FUNC(add4st1)
	saveXmm6
	saveXmm7
	addBegin st=%xmm7
	shl      $5,r_lo
	shl      $5,r_hi
	movdqa   (r_bb,r_lo,1),%xmm5
	movdqa   (r_bb,r_hi,1),%xmm6
	pcmpeqq  %xmm0,%xmm7
	movdqa   %xmm5,%xmm1
	pcmpeqq  %xmm0,%xmm5
	por      %xmm7,%xmm5
	movdqa   16(r_bb,r_lo,1),%xmm7
	movdqa   %xmm6,%xmm2
	pcmpeqq  %xmm0,%xmm6
	movdqa   %xmm7,%xmm3
	pcmpeqq  %xmm0,%xmm7
	por      %xmm7,%xmm6
	movdqa   16(r_bb,r_hi,1),%xmm7
	por      %xmm6,%xmm5
	movdqa   %xmm7,%xmm4
	pcmpeqq  %xmm0,%xmm7
	por      %xmm7,%xmm5
	restoreXmm7
	addEnd   %xmm5 NAME(insert4) usingXmm6=1
END(add4st1)

FUNC(has4st1)
	hasBegin st=%xmm3
	shl      $5,r_lo
	shl      $5,r_hi
	movdqa   (r_bb,r_lo,1),%xmm1
	movdqa   (r_bb,r_hi,1),%xmm2
	pcmpeqq  %xmm0,%xmm3
	pcmpeqq  %xmm0,%xmm1
	por      %xmm3,%xmm1
	movdqa   16(r_bb,r_lo,1),%xmm3
	pcmpeqq  %xmm0,%xmm2
	pcmpeqq  %xmm0,%xmm3
	pcmpeqq  16(r_bb,r_hi,1),%xmm0
	por      %xmm3,%xmm2
	por      %xmm2,%xmm1
	por      %xmm0,%xmm1
	hasEnd   %xmm1
END(has4st1)
