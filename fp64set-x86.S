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

#define NAME(name) fp64set_##name
#define FPTR(name) $NAME(name)

#define FUNC(name)  \
	.global    NAME(name); \
	.hidden    NAME(name); \
	.type      NAME(name),@function; \
	.align     16; \
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

FUNC(has2st0sse4)
	REG3
	movd     r_lo,%XMM0
	pinsrd   $1,r_hi,%XMM0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	mov      m_bb(r_ptr),r_ptr
	movddup  %XMM0,%XMM0
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm1
	pcmpeqq  %XMM0,%xmm1
	pcmpeqq  (r_ptr,r_hi,1),%XMM0
	por      %XMM0,%xmm1
	pmovmskb %xmm1,%eax
	RET
END(has2st0sse4)

FUNC(add2st0sse4)
	REG3
	ALLOC
	SAVE_PTR
	movd     r_lo,%xmm0
	pinsrd   $1,r_hi,%xmm0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	mov      m_bb(r_ptr),r_ptr
	movddup  %xmm0,%xmm0
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm3
	movdqa   (r_ptr,r_hi,1),%xmm4
	movdqa   %xmm3,%xmm1
	pcmpeqq  %xmm0,%xmm3
	movdqa   %xmm4,%xmm2
	pcmpeqq  %xmm0,%xmm4
	por      %xmm4,%xmm3
	pmovmskb %xmm3,r_ptr
	test     r_ptr,r_ptr
	jz       NAME(insert2sse4)
	FREE
	xor      %eax,%eax
	RET
END(add2st0sse4)

FUNC(insert2sse4)
	// Strategically bump set->cnt.
	mov      save_ptr(%esp),r_ptr
	addl     $1,m_cnt(r_ptr)
	mov      m_bb(r_ptr),r_ptr
	mov      %ebx,save_ebx(%esp)

	// Blank values for b1[*] -> xmm3.
	movd     r_lo,%xmm3
	pxor     %xmm5,%xmm5
	movddup  %xmm3,%xmm3
	pcmpeqq  %xmm5,%xmm3

	// Check for free slots in b1[*] -> xmm3.
	pcmpeqq  %xmm1,%xmm3

	// Blank values for b2[*] -> xmm4.
	movd     r_hi,%xmm4
	movddup  %xmm4,%xmm4
	pcmpeqq  %xmm2,%xmm4

	// Check for free slots in b2[*] -> xmm4.
	pcmpeqq  %xmm2,%xmm4

	// Combine (b1[0],b2[0],b1[1],b2[1]) -> mask.
	pblendw  $0xcc,%xmm4,%xmm3
	pmovmskb %xmm3,%ebx
	test     %ebx,%ebx
	jnz      1f

	// Nope, preparing for the kickAdd() loop.
	mov      %esi,save_esi(%esp)
	mov      %edi,save_edi(%esp)
	mov      save_ptr(%esp),%ebx
	mov      m_mask(%ebx),%esi
	mov      m_logsize(%ebx),%edi
	shl      $1,%edi
	jmp      2f

	// Successful justAdd() lands here.
1:	bsf      %ebx,%ebx
	// Is it b1[] or b2[]? bsf returns 0, 4, 8, or 12.
	test     $4,%ebx
	cmovnz   r_hi,r_lo
	// Is it [0] or [1]?
	shr      $3,%ebx
	lea      (r_lo,%ebx,8),r_lo
	mov      save_ebx(%esp),%ebx
	FREE
	movq     %xmm0,(r_ptr,r_lo,1)
	mov      $1,%eax
	RET

	// Loop: the element in xmm0 is pushed at eax from the top.
	// The element kicked out from the bottom is in xmm1.
	.align   32
2:	palignr  $8,%xmm1,%xmm0
	movdqa   %xmm0,(r_ptr,r_lo,1)

	// Find the alternative bucket for xmm1 at r_hi.
	movd     %xmm1,r_hi
	pextrd   $1,%xmm1,%ebx
	and      %esi,r_hi
	and      %esi,%ebx
	shl      $4,r_hi
	shl      $4,%ebx
	cmp      r_lo,r_hi
	cmove    %ebx,r_hi

	// Make the blank value for bb[r_hi].
	movd     r_hi,%xmm3
	pinsrd   $2,r_hi,%xmm3
	pcmpeqq  %xmm5,%xmm3

	// Check for empty slots and insert.
	movdqa   (r_ptr,r_hi,1),%xmm2
	pcmpeqq  %xmm2,%xmm3
	pmovmskb %xmm3,%ebx
	test     %ebx,%ebx
	jnz      3f

	// No free slot at r_hi.
	movq     %xmm1,%xmm0
	movdqa   %xmm2,%xmm1
	mov      r_hi,r_lo

	// Loop control.
	sub      $1,%edi
	jge      2b

	// Broke out of the loop.  Too bad.
	mov      save_ptr(%esp),r_ptr
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	FREE
	movd     %xmm0,r_lo
	pextrd   $1,%xmm0,r_hi
	jmp      NAME(insert2tail)

	// Successful kickAdd() lands here.
3:	bsf      %ebx,%ebx
	add      %ebx,r_hi
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	movq     %xmm1,(r_ptr,r_hi,1)
	FREE
	mov      $1,%eax
	RET
END(insert2sse4)

FUNC(add2st1sse4)
	REG3
	ALLOC
	SAVE_PTR
	movd     r_lo,%xmm0
	pinsrd   $1,r_hi,%xmm0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	movdqa   m_stash(r_ptr),%xmm5
	movddup  %xmm0,%xmm0
	mov      m_bb(r_ptr),r_ptr
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm3
	movdqa   (r_ptr,r_hi,1),%xmm4
	pcmpeqq  %xmm0,%xmm5
	movdqa   %xmm3,%xmm1
	pcmpeqq  %xmm0,%xmm3
	por      %xmm5,%xmm3
	movdqa   %xmm4,%xmm2
	pcmpeqq  %xmm0,%xmm4
	por      %xmm4,%xmm3
	pmovmskb %xmm3,r_ptr
	test     r_ptr,r_ptr
	jz       NAME(insert2sse4)
	FREE
	xor      %eax,%eax
	RET
END(add2st1sse4)

FUNC(has2st1sse4)
	REG3
	movd     r_lo,%XMM0
	pinsrd   $1,r_hi,%XMM0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	movdqa   m_stash(r_ptr),%xmm3
	mov      m_bb(r_ptr),r_ptr
	movddup  %XMM0,%XMM0
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm1
	pcmpeqq  %XMM0,%xmm3
	pcmpeqq  %XMM0,%xmm1
	pcmpeqq  (r_ptr,r_hi,1),%XMM0
	por      %xmm3,%xmm1
	por      %XMM0,%xmm1
	pmovmskb %xmm1,%eax
	RET
END(has2st1sse4)

FUNC(has3st0sse4)
	REG3
	movd     r_lo,%XMM0
	pinsrd   $1,r_hi,%XMM0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	mov      m_bb(r_ptr),r_ptr
	lea      (r_lo,r_lo,2),r_lo
	lea      (r_hi,r_hi,2),r_hi
	movddup  %XMM0,%XMM0
	movdqu   8(r_ptr,r_lo,8),%xmm1
	movdqu   8(r_ptr,r_hi,8),%xmm2
	pcmpeqq  %XMM0,%xmm1
	movq     (r_ptr,r_lo,8),%xmm3
	movhps   (r_ptr,r_hi,8),%xmm3
	pcmpeqq  %XMM0,%xmm2
	pcmpeqq  %XMM0,%xmm3
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	pmovmskb %xmm1,%eax
	RET
END(has3st0sse4)

FUNC(add3st0sse4)
	REG3
	ALLOC
	SAVE_PTR
	movd     r_lo,%xmm0
	pinsrd   $1,r_hi,%xmm0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	mov      m_bb(r_ptr),r_ptr
	lea      (r_lo,r_lo,2),r_lo
	lea      (r_hi,r_hi,2),r_hi
	movddup  %xmm0,%xmm0
	movdqu   (r_ptr,r_lo,8),%xmm4
	movdqu   (r_ptr,r_hi,8),%xmm5
	movq     16(r_ptr,r_lo,8),%xmm3
	movhps   16(r_ptr,r_hi,8),%xmm3
	movdqa   %xmm4,%xmm1
	pcmpeqq  %xmm0,%xmm4
	movdqa   %xmm5,%xmm2
	pcmpeqq  %xmm0,%xmm5
	por      %xmm5,%xmm4
	movdqa   %xmm3,%xmm5
	pcmpeqq  %xmm0,%xmm5
	por      %xmm5,%xmm4
	pmovmskb %xmm4,r_ptr
	test     r_ptr,r_ptr
	jz       NAME(insert3sse4)
	FREE
	xor      %eax,%eax
	RET
END(add3st0sse4)

FUNC(insert3sse4)
	// Strategically bump set->cnt.
	mov      save_ptr(%esp),r_ptr
	addl     $1,m_cnt(r_ptr)
	mov      m_bb(r_ptr),r_ptr
	mov      %ebx,save_ebx(%esp)

	// Blank values for b2[*] -> xmm6.
	movd     r_hi,%xmm6
	pxor     %xmm5,%xmm5
	movddup  %xmm6,%xmm6
	pcmpeqq  %xmm5,%xmm6

	// Check for free slots in b2[0,1] -> xmm2.
	pcmpeqq  %xmm6,%xmm2

	// Blank values for b1[*] -> xmm4.
	movd     r_lo,%xmm4
	movddup  %xmm4,%xmm4
	pcmpeqq  %xmm5,%xmm4

	// Blank values for b1[*] + b2[*] -> xmm6.
	pblendw  $0x0f,%xmm4,%xmm6

	// Check for free slots in b1[0,1] -> xmm4.
	pcmpeqq  %xmm1,%xmm4

	// Combine (b1[0],b2[0],b1[1],b2[1]) -> xmm4.
	pblendw  $0xcc,%xmm2,%xmm4

	// Check for free slots in b1[2],b2[2] -> xmm6.
	pcmpeqq  %xmm3,%xmm6

	// Recombine (b1[0],b2[0],b1[1],b2[1],b1[2],b2[2],b1[2],b2[2]) -> mask.
	pshufd   $0xd8,%xmm6,%xmm6
	packssdw %xmm6,%xmm4
	pmovmskb %xmm4,%ebx
	test     %ebx,%ebx
	jnz      1f

	// Nope, preparing for the kickAdd() loop.
	mov      %esi,save_esi(%esp)
	mov      %edi,save_edi(%esp)
	mov      save_ptr(%esp),%ebx
	mov      m_mask(%ebx),%esi
	mov      m_logsize(%ebx),%edi
	shl      $1,%edi
	jmp      2f

	// Successful justAdd() lands here.
1:	bsf      %ebx,%ebx
	// Is it b1[] or b2[]? bsf returns (0, 4, 8) vs (2, 6, 10).
	test     $2,%ebx
	cmovnz   r_hi,r_lo
	// Is it [0], [1] or [2]?
	shr      $2,%ebx
	add      %ebx,r_lo
	mov      save_ebx(%esp),%ebx
	FREE
	movq     %xmm0,(r_ptr,r_lo,8)
	mov      $1,%eax
	RET

	// Loop: the element in xmm0 is pushed at eax from the top.
	// The element kicked out from the bottom is in xmm1.
	.align   32
2:	palignr  $8,%xmm1,%xmm3
	movdqu   %xmm3,(r_ptr,r_lo,8)
	movq     %xmm0,16(r_ptr,r_lo,8)

	// Find the alternative bucket for xmm1 at r_hi.
	movd     %xmm1,r_hi
	pextrd   $1,%xmm1,%ebx
	and      %esi,r_hi
	and      %esi,%ebx
	lea      (r_hi,r_hi,2),r_hi
	lea      (%ebx,%ebx,2),%ebx
	cmp      r_lo,r_hi
	cmove    %ebx,r_hi

	// Make the blank value for bb[r_hi].
	movd     r_hi,%xmm4
	pinsrd   $2,r_hi,%xmm4
	pcmpeqq  %xmm5,%xmm4

	// Check for empty slots and insert.
	movdqu   (r_ptr,r_hi,8),%xmm2
	movq     16(r_ptr,r_hi,8),%xmm3
	movdqa   %xmm2,%xmm6
	pcmpeqq  %xmm4,%xmm6
	pcmpeqq  %xmm3,%xmm4
	packssdw %xmm4,%xmm6
	pmovmskb %xmm6,%ebx
	and      $0x0fff,%ebx
	jnz      3f

	// No free slot at r_hi.
	movq     %xmm1,%xmm0
	movdqa   %xmm2,%xmm1
	mov      r_hi,r_lo

	// Loop control.
	sub      $1,%edi
	jge      2b

	// Broke out of the loop.  Too bad.
	mov      save_ptr(%esp),r_ptr
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	FREE
	movd     %xmm0,r_lo
	pextrd   $1,%xmm0,r_hi
	jmp      NAME(insert3tail)

	// Successful kickAdd() lands here.
3:	bsf      %ebx,%ebx
	shr      $2,%ebx
	add      %ebx,r_hi
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	movq     %xmm1,(r_ptr,r_hi,8)
	FREE
	mov      $1,%eax
	RET
END(insert3sse4)

FUNC(add3st1sse4)
	REG3
	ALLOC
	SAVE_PTR
	movd     r_lo,%xmm0
	pinsrd   $1,r_hi,%xmm0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	movdqa   m_stash(r_ptr),%xmm5
	mov      m_bb(r_ptr),r_ptr
	movddup  %xmm0,%xmm0
	lea      (r_lo,r_lo,2),r_lo
	lea      (r_hi,r_hi,2),r_hi
	movdqu   (r_ptr,r_lo,8),%xmm4
	pcmpeqq  %xmm0,%xmm5
	movdqa   %xmm4,%xmm1
	pcmpeqq  %xmm0,%xmm4
	por      %xmm5,%xmm4
	movdqu   (r_ptr,r_hi,8),%xmm5
	movdqa   %xmm5,%xmm2
	pcmpeqq  %xmm0,%xmm5
	por      %xmm5,%xmm4
	movq     16(r_ptr,r_lo,8),%xmm5
	movhps   16(r_ptr,r_hi,8),%xmm5
	movdqa   %xmm5,%xmm3
	pcmpeqq  %xmm0,%xmm5
	por      %xmm5,%xmm4
	pmovmskb %xmm4,r_ptr
	test     r_ptr,r_ptr
	jz       NAME(insert3sse4)
	FREE
	xor      %eax,%eax
	RET
END(add3st1sse4)

FUNC(has3st1sse4)
	REG3
	movd     r_lo,%XMM0
	pinsrd   $1,r_hi,%XMM0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	movdqa   m_stash(r_ptr),%xmm3
	mov      m_bb(r_ptr),r_ptr
	movddup  %XMM0,%XMM0
	lea      (r_lo,r_lo,2),r_lo
	lea      (r_hi,r_hi,2),r_hi
	movdqu   8(r_ptr,r_lo,8),%xmm1
	movdqu   8(r_ptr,r_hi,8),%xmm2
	pcmpeqq  %XMM0,%xmm3
	pcmpeqq  %XMM0,%xmm1
	pcmpeqq  %XMM0,%xmm2
	por      %xmm3,%xmm1
	movq     (r_ptr,r_lo,8),%xmm3
	movhps   (r_ptr,r_hi,8),%xmm3
	pcmpeqq  %XMM0,%xmm3
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	pmovmskb %xmm1,%eax
	RET
END(has3st1sse4)

FUNC(has4st0sse4)
	REG3
	movd     r_lo,%XMM0
	pinsrd   $1,r_hi,%XMM0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	mov      m_bb(r_ptr),r_ptr
	movddup  %XMM0,%XMM0
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm1
	movdqa   (r_ptr,r_hi,1),%xmm2
	pcmpeqq  %XMM0,%xmm1
	movdqa   16(r_ptr,r_lo,1),%xmm3
	pcmpeqq  %XMM0,%xmm2
	pcmpeqq  %XMM0,%xmm3
	pcmpeqq  16(r_ptr,r_hi,1),%XMM0
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	por      %XMM0,%xmm1
	pmovmskb %xmm1,%eax
	RET
END(has4st0sse4)

FUNC(add4st0sse4)
	REG3
	ALLOC
	SAVE_PTR
	movd     r_lo,%xmm0
	pinsrd   $1,r_hi,%xmm0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	mov      m_bb(r_ptr),r_ptr
	movddup  %xmm0,%xmm0
	shl      $5,r_lo
	shl      $5,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm5
	movdqa   (r_ptr,r_hi,1),%xmm6
	movdqa   %xmm5,%xmm1
	pcmpeqq  %xmm0,%xmm5
	movdqa   %xmm6,%xmm2
	pcmpeqq  %xmm0,%xmm6
	por      %xmm6,%xmm5
	movdqa   16(r_ptr,r_lo,1),%xmm6
	movdqa   16(r_ptr,r_hi,1),%xmm7
	movdqa   %xmm6,%xmm3
	pcmpeqq  %xmm0,%xmm6
	movdqa   %xmm7,%xmm4
	pcmpeqq  %xmm0,%xmm7
	por      %xmm7,%xmm6
	por      %xmm6,%xmm5
	pmovmskb %xmm5,r_ptr
	test     r_ptr,r_ptr
	jz       NAME(insert4sse4)
	FREE
	xor      %eax,%eax
	RET
END(add4st0sse4)

FUNC(insert4sse4)
	// Strategically bump set->cnt.
	mov      save_ptr(%esp),r_ptr
	addl     $1,m_cnt(r_ptr)
	mov      m_bb(r_ptr),r_ptr
	mov      %ebx,save_ebx(%esp)

	// Blank values for b2[*] -> xmm6.
	movd     r_hi,%xmm6
	pxor     %xmm5,%xmm5
	movddup  %xmm6,%xmm6
	pcmpeqq  %xmm5,%xmm6

	// Check for free slots in b2[*] -> xmm2.
	pcmpeqq  %xmm6,%xmm2
	pcmpeqq  %xmm6,%xmm4
	packssdw %xmm4,%xmm2

	// Blank values for b1[*] -> xmm6.
	movd     r_lo,%xmm6
	movddup  %xmm6,%xmm6
	pcmpeqq  %xmm5,%xmm6

	// Check for free slots in b1[*] -> xmm6.
	movdqa   %xmm3,%xmm4
	pcmpeqq  %xmm6,%xmm4
	pcmpeqq  %xmm1,%xmm6
	packssdw %xmm4,%xmm6

	// Recombine (b1[0],b2[0],b1[1],b2[1],b1[2],b2[2],b1[3],b2[3]) -> mask.
	pblendw  $0xaa,%xmm2,%xmm6
	pmovmskb %xmm6,%ebx
	test     %ebx,%ebx
	jnz      1f

	// Nope, preparing for the kickAdd() loop.
	mov      %esi,save_esi(%esp)
	mov      %edi,save_edi(%esp)
	mov      save_ptr(%esp),%ebx
	mov      m_mask(%ebx),%esi
	mov      m_logsize(%ebx),%edi
	shl      $1,%edi
	jmp      2f

	// Successful justAdd() lands here.
1:	bsf      %ebx,%ebx
	// Is it b1[] or b2[]? bsf returns (0, 4, 8, 12) vs (2, 6, 10, 14).
	test     $2,%ebx
	cmovnz   r_hi,r_lo
	// Is it [0], [1], [2], or [3]?
	shr      $2,%ebx
	lea      (r_lo,%ebx,8),r_lo
	mov      save_ebx(%esp),%ebx
	FREE
	movq     %xmm0,(r_ptr,r_lo,1)
	mov      $1,%eax
	RET

	// Loop: the element in xmm0 is pushed at eax from the top.
	// The element kicked out from the bottom is in xmm1.
	.align   32
2:	palignr  $8,%xmm3,%xmm0
	movdqa   %xmm0,16(r_ptr,r_lo,1)
	palignr  $8,%xmm1,%xmm3
	movdqa   %xmm3,(r_ptr,r_lo,1)

	// Find the alternative bucket for xmm1 at r_hi.
	movd     %xmm1,r_hi
	pextrd   $1,%xmm1,%ebx
	and      %esi,r_hi
	and      %esi,%ebx
	shl      $5,r_hi
	shl      $5,%ebx
	cmp      r_lo,r_hi
	cmove    %ebx,r_hi

	// Make the blank value for bb[r_hi].
	movd     r_hi,%xmm4
	pinsrd   $2,r_hi,%xmm4
	pcmpeqq  %xmm5,%xmm4

	// Check for empty slots and insert.
	movdqa   (r_ptr,r_hi,1),%xmm2
	movdqa   16(r_ptr,r_hi,1),%xmm3
	movdqa   %xmm2,%xmm6
	pcmpeqq  %xmm4,%xmm6
	pcmpeqq  %xmm3,%xmm4
	packssdw %xmm4,%xmm6
	pmovmskb %xmm6,%ebx
	test     %ebx,%ebx
	jnz      3f

	// No free slot at r_hi.
	movq     %xmm1,%xmm0
	movdqa   %xmm2,%xmm1
	mov      r_hi,r_lo

	// Loop control.
	sub      $1,%edi
	jge      2b

	// Broke out of the loop.  Too bad.
	mov      save_ptr(%esp),r_ptr
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	FREE
	movd     %xmm0,r_lo
	pextrd   $1,%xmm0,r_hi
	jmp      NAME(insert4tail)

	// Successful kickAdd() lands here.
3:	bsf      %ebx,%ebx
	lea      (r_hi,%ebx,2),r_hi
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	movq     %xmm1,(r_ptr,r_hi,1)
	FREE
	mov      $1,%eax
	RET
END(insert4sse4)

FUNC(add4st1sse4)
	REG3
	ALLOC
	SAVE_PTR
	movd     r_lo,%xmm0
	pinsrd   $1,r_hi,%xmm0
	movdqa   m_stash(r_ptr),%xmm7
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	mov      m_bb(r_ptr),r_ptr
	movddup  %xmm0,%xmm0
	shl      $5,r_lo
	shl      $5,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm5
	movdqa   (r_ptr,r_hi,1),%xmm6
	pcmpeqq  %xmm0,%xmm7
	movdqa   %xmm5,%xmm1
	pcmpeqq  %xmm0,%xmm5
	movdqa   %xmm6,%xmm2
	pcmpeqq  %xmm0,%xmm6
	por      %xmm7,%xmm6
	por      %xmm6,%xmm5
	movdqa   16(r_ptr,r_lo,1),%xmm6
	movdqa   16(r_ptr,r_hi,1),%xmm7
	movdqa   %xmm6,%xmm3
	pcmpeqq  %xmm0,%xmm6
	movdqa   %xmm7,%xmm4
	pcmpeqq  %xmm0,%xmm7
	por      %xmm7,%xmm6
	por      %xmm6,%xmm5
	pmovmskb %xmm5,r_ptr
	test     r_ptr,r_ptr
	jz       NAME(insert4sse4)
	FREE
	xor      %eax,%eax
	RET
END(add4st1sse4)

FUNC(has4st1sse4)
	REG3
	movd     r_lo,%XMM0
	pinsrd   $1,r_hi,%XMM0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	movdqa   m_stash(r_ptr),%xmm3
	mov      m_bb(r_ptr),r_ptr
	movddup  %XMM0,%XMM0
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm1
	movdqa   (r_ptr,r_hi,1),%xmm2
	pcmpeqq  %XMM0,%xmm3
	pcmpeqq  %XMM0,%xmm1
	por      %xmm3,%xmm1
	movdqa   16(r_ptr,r_lo,1),%xmm3
	pcmpeqq  %XMM0,%xmm2
	pcmpeqq  %XMM0,%xmm3
	pcmpeqq  16(r_ptr,r_hi,1),%XMM0
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	por      %XMM0,%xmm1
	pmovmskb %xmm1,%eax
	RET
END(has4st1sse4)
