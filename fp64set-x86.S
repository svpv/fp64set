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
#define HAS_REGS   mov 4(%esp),r_ptr
#define HAS_RET    ret $4
#else // REGPARM3
#define r_lo       %eax
#define r_hi       %edx
#define r_ptr      %ecx
#define HAS_REGS
#define HAS_RET    ret
#endif

FUNC(has2st0sse4)
	HAS_REGS
	movd     r_lo,%xmm0
	pinsrd   $1,r_hi,%xmm0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	mov      m_bb(r_ptr),r_ptr
	movddup  %xmm0,%xmm0
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm1
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  (r_ptr,r_hi,1),%xmm0
	por      %xmm0,%xmm1
	pmovmskb %xmm1,%eax
	HAS_RET
END(has2st0sse4)

FUNC(add2st0sse4)
	sub      $16,%esp
#define save_ecx 0
#define save_ebx 4
#define save_esi 8
#define save_edi 12
	mov      %ecx,save_ecx(%esp)
	movd     %eax,%xmm0
	pinsrd   $1,%edx,%xmm0
	and      m_mask(%ecx),%eax
	and      m_mask(%ecx),%edx
	mov      m_bb(%ecx),%ecx
	movddup  %xmm0,%xmm0
	shl      $4,%eax
	shl      $4,%edx
	movdqa   (%ecx,%eax,1),%xmm1
	movdqa   (%ecx,%edx,1),%xmm2
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	por      %xmm2,%xmm1
	pmovmskb %xmm1,%ecx
	test     %ecx,%ecx
	jz       NAME(insert2sse4)
	add      $16,%esp
	xor      %eax,%eax
	ret
END(add2st0sse4)

FUNC(insert2sse4)
	// Strategically bump set->cnt.
	mov      save_ecx(%esp),%ecx
	addl     $1,m_cnt(%ecx)
	mov      m_bb(%ecx),%ecx
	mov      %ebx,save_ebx(%esp)

	// Blank values for b1[*] + b2[*] -> xmm3.
	movd     %eax,%xmm3
	pinsrd   $2,%edx,%xmm3
	pxor     %xmm6,%xmm6
	pcmpeqq  %xmm6,%xmm3

	// Load b1[0] + b2[0], check for a free slot.
	movq     (%ecx,%eax,1),%xmm1
	movhps   (%ecx,%edx,1),%xmm1
	pcmpeqq  %xmm3,%xmm1
	pmovmskb %xmm1,%ebx
	test     %ebx,%ebx
	jnz      2f

	// Load b1[1] + b2[1], check for a free slot.
	movq     8(%ecx,%eax,1),%xmm1
	movhps   8(%ecx,%edx,1),%xmm1
	pcmpeqq  %xmm3,%xmm1
	pmovmskb %xmm1,%ebx
	test     %ebx,%ebx
	jnz      3f

	// Nope, preparing for the kickAdd() loop.
	mov      %esi,save_esi(%esp)
	mov      %edi,save_edi(%esp)
	mov      save_ecx(%esp),%ebx
	mov      m_logsize(%ebx),%esi
	shl      $1,%esi
	mov      m_mask(%ebx),%ebx
	jmp      10f

	// Successful justAdd() lands here.
2:	test     %bl,%bl
	cmovz    %edx,%eax
	mov      save_ebx(%esp),%ebx
	add      $16,%esp
	movq     %xmm0,(%ecx,%eax,1)
	mov      $1,%eax
	ret
3:	test     %bl,%bl
	cmovz    %edx,%eax
	mov      save_ebx(%esp),%ebx
	add      $16,%esp
	movq     %xmm0,8(%ecx,%eax,1)
	mov      $1,%eax
	ret

	// Loop: the element in xmm0 is pushed at eax from the top.
	// The element kicked out from the bottom goes into xmm7.
	.align   32
10:	movq     (%ecx,%eax,1),%xmm7
	movq     8(%ecx,%eax,1),%xmm1
	movq     %xmm1,(%ecx,%eax,1)
	movq     %xmm0,8(%ecx,%eax,1)

	// Find the alternative bucket for xmm7 at edx.
	movd     %xmm7,%edx
	pextrd   $1,%xmm7,%edi
	and      %ebx,%edx
	and      %ebx,%edi
	shl      $4,%edx
	shl      $4,%edi
	cmp      %eax,%edx
	cmovz    %edi,%edx

	// Make the blank value for bb[edx].
	movd     %edx,%xmm1
	pinsrd   $2,%edx,%xmm1
	pcmpeqq  %xmm6,%xmm1

	// Check for empty slots and insert.
	pcmpeqq  (%ecx,%edx,1),%xmm1
	pmovmskb %xmm1,%eax
	test     %eax,%eax
	jnz      12f

	// No free slot at edx.
	movdqa   %xmm7,%xmm0
	mov      %edx,%eax

	// Loop control.
	sub      $1,%esi
	jge      10b

	// Broke out of the loop.  Too bad.
	mov      save_ecx(%esp),%ecx
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	add      $16,%esp
	movd     %xmm0,%eax
	pextrd   $1,%xmm0,%edx
	jmp      NAME(insert2tail)

	// Successful kickAdd() lands here.
12:	bsf      %eax,%eax
	add      %eax,%edx
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	movq     %xmm7,(%ecx,%edx,1)
	add      $16,%esp
	mov      $1,%eax
	ret
END(insert2sse4)

FUNC(add2st1sse4)
	sub      $16,%esp
	mov      %ecx,save_ecx(%esp)
	movd     %eax,%xmm0
	pinsrd   $1,%edx,%xmm0
	and      m_mask(%ecx),%eax
	and      m_mask(%ecx),%edx
	movdqa   m_stash(%ecx),%xmm7
	movddup  %xmm0,%xmm0
	mov      m_bb(%ecx),%ecx
	shl      $4,%eax
	shl      $4,%edx
	movdqa   (%ecx,%eax,1),%xmm1
	movdqa   (%ecx,%edx,1),%xmm2
	pcmpeqq  %xmm0,%xmm7
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	por      %xmm7,%xmm1
	por      %xmm2,%xmm1
	pmovmskb %xmm1,%ecx
	test     %ecx,%ecx
	jz       NAME(insert2sse4)
	add      $16,%esp
	xor      %eax,%eax
	ret
END(add2st1sse4)

FUNC(has2st1sse4)
	HAS_REGS
	movd     r_lo,%xmm0
	pinsrd   $1,r_hi,%xmm0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	movdqa   m_stash(r_ptr),%xmm7
	movddup  %xmm0,%xmm0
	mov      m_bb(r_ptr),r_ptr
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm1
	pcmpeqq  %xmm0,%xmm7
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  (r_ptr,r_hi,1),%xmm0
	por      %xmm7,%xmm0
	por      %xmm0,%xmm1
	pmovmskb %xmm1,%eax
	HAS_RET
END(has2st1sse4)

FUNC(has3st0sse4)
	HAS_REGS
	movd     r_lo,%xmm0
	pinsrd   $1,r_hi,%xmm0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	mov      m_bb(r_ptr),r_ptr
	lea      (r_lo,r_lo,2),r_lo
	lea      (r_hi,r_hi,2),r_hi
	movddup  %xmm0,%xmm0
	movdqu   8(r_ptr,r_lo,8),%xmm1
	movdqu   8(r_ptr,r_hi,8),%xmm2
	movq     (r_ptr,r_lo,8),%xmm3
	movhps   (r_ptr,r_hi,8),%xmm3
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	pcmpeqq  %xmm0,%xmm3
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	pmovmskb %xmm1,%eax
	HAS_RET
END(has3st0sse4)

FUNC(add3st0sse4)
	sub      $16,%esp
	mov      %ecx,save_ecx(%esp)
	movd     %eax,%xmm0
	pinsrd   $1,%edx,%xmm0
	and      m_mask(%ecx),%eax
	and      m_mask(%ecx),%edx
	mov      m_bb(%ecx),%ecx
	lea      (%eax,%eax,2),%eax
	lea      (%edx,%edx,2),%edx
	movddup  %xmm0,%xmm0
	movdqu   8(%ecx,%eax,8),%xmm1
	movdqu   8(%ecx,%edx,8),%xmm2
	movq     (%ecx,%eax,8),%xmm3
	movhps   (%ecx,%edx,8),%xmm3
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	pcmpeqq  %xmm0,%xmm3
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	pmovmskb %xmm1,%ecx
	test     %ecx,%ecx
	jz       NAME(insert3sse4)
	add      $16,%esp
	xor      %eax,%eax
	ret
END(add3st0sse4)

FUNC(insert3sse4)
	// Strategically bump set->cnt.
	mov      save_ecx(%esp),%ecx
	addl     $1,m_cnt(%ecx)
	mov      m_bb(%ecx),%ecx
	mov      %ebx,save_ebx(%esp)

	// Blank values for b1[*] + b2[*] -> xmm3.
	movd     %eax,%xmm3
	pinsrd   $2,%edx,%xmm3
	pxor     %xmm6,%xmm6
	pcmpeqq  %xmm6,%xmm3

	// Load b1[0] + b2[0], check for a free slot.
	movq     (%ecx,%eax,8),%xmm1
	movhps   (%ecx,%edx,8),%xmm1
	pcmpeqq  %xmm3,%xmm1
	pmovmskb %xmm1,%ebx
	test     %ebx,%ebx
	jnz      2f

	// Load b1[1] + b2[1], check for a free slot.
	movq     8(%ecx,%eax,8),%xmm1
	movhps   8(%ecx,%edx,8),%xmm1
	pcmpeqq  %xmm3,%xmm1
	pmovmskb %xmm1,%ebx
	test     %ebx,%ebx
	jnz      3f

	// Load b1[2] + b2[2], check for a free slot.
	movq     16(%ecx,%eax,8),%xmm1
	movhps   16(%ecx,%edx,8),%xmm1
	pcmpeqq  %xmm3,%xmm1
	pmovmskb %xmm1,%ebx
	test     %ebx,%ebx
	jnz      4f

	// Nope, preparing for the kickAdd() loop.
	mov      %esi,save_esi(%esp)
	mov      %edi,save_edi(%esp)
	mov      save_ecx(%esp),%ebx
	mov      m_logsize(%ebx),%esi
	shl      $1,%esi
	mov      m_mask(%ebx),%ebx
	jmp      10f

	// Successful justAdd() lands here.
2:	test     %bl,%bl
	cmovz    %edx,%eax
	mov      save_ebx(%esp),%ebx
	add      $16,%esp
	movq     %xmm0,(%ecx,%eax,8)
	mov      $1,%eax
	ret
3:	test     %bl,%bl
	cmovz    %edx,%eax
	mov      save_ebx(%esp),%ebx
	add      $16,%esp
	movq     %xmm0,8(%ecx,%eax,8)
	mov      $1,%eax
	ret
4:	test     %bl,%bl
	cmovz    %edx,%eax
	mov      save_ebx(%esp),%ebx
	add      $16,%esp
	movq     %xmm0,16(%ecx,%eax,8)
	mov      $1,%eax
	ret

	// Loop: the element in xmm0 is pushed at eax from the top.
	// The element kicked out from the bottom goes into xmm7.
	.align   32
10:	movq     (%ecx,%eax,8),%xmm7
	movdqu   8(%ecx,%eax,8),%xmm1
	movdqu   %xmm1,(%ecx,%eax,8)
	movq     %xmm0,16(%ecx,%eax,8)

	// Find the alternative bucket for xmm7 at edx.
	movd     %xmm7,%edx
	pextrd   $1,%xmm7,%edi
	and      %ebx,%edx
	and      %ebx,%edi
	lea      (%edx,%edx,2),%edx
	lea      (%edi,%edi,2),%edi
	cmp      %eax,%edx
	cmovz    %edi,%edx

	// Make the blank value for bb[edx].
	movd     %edx,%xmm1
	pinsrd   $2,%edx,%xmm1
	pcmpeqq  %xmm6,%xmm1

	// Check for empty slots and insert.
	movdqu   8(%ecx,%edx,8),%xmm3
	movdqu   (%ecx,%edx,8),%xmm2
	pcmpeqq  %xmm1,%xmm3
	pcmpeqq  %xmm1,%xmm2
	pmovmskb %xmm3,%eax
	pmovmskb %xmm2,%edi
	shl      $8,%eax
	or       %edi,%eax
	jnz      12f

	// No free slot at edx.
	movdqa   %xmm7,%xmm0
	mov      %edx,%eax

	// Loop control.
	sub      $1,%esi
	jge      10b

	// Broke out of the loop.  Too bad.
	mov      save_ecx(%esp),%ecx
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	add      $16,%esp
	movd     %xmm0,%eax
	pextrd   $1,%xmm0,%edx
	jmp      NAME(insert3tail)

	// Successful kickAdd() lands here.
12:	bsf      %eax,%eax
	shr      $3,%eax
	add      %eax,%edx
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	movq     %xmm7,(%ecx,%edx,8)
	add      $16,%esp
	mov      $1,%eax
	ret
END(insert3sse4)

FUNC(add3st1sse4)
	sub      $16,%esp
	mov      %ecx,save_ecx(%esp)
	movd     %eax,%xmm0
	pinsrd   $1,%edx,%xmm0
	and      m_mask(%ecx),%eax
	and      m_mask(%ecx),%edx
	movdqa   m_stash(%ecx),%xmm7
	movddup  %xmm0,%xmm0
	mov      m_bb(%ecx),%ecx
	lea      (%eax,%eax,2),%eax
	lea      (%edx,%edx,2),%edx
	movdqu   8(%ecx,%eax,8),%xmm1
	movdqu   8(%ecx,%edx,8),%xmm2
	pcmpeqq  %xmm0,%xmm7
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	por      %xmm7,%xmm1
	movq     (%ecx,%eax,8),%xmm3
	movhps   (%ecx,%edx,8),%xmm3
	pcmpeqq  %xmm0,%xmm3
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	pmovmskb %xmm1,%ecx
	test     %ecx,%ecx
	jz       NAME(insert3sse4)
	add      $16,%esp
	xor      %eax,%eax
	ret
END(add3st1sse4)

FUNC(has3st1sse4)
	HAS_REGS
	movd     r_lo,%xmm0
	pinsrd   $1,r_hi,%xmm0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	movdqa   m_stash(r_ptr),%xmm7
	movddup  %xmm0,%xmm0
	mov      m_bb(r_ptr),r_ptr
	lea      (r_lo,r_lo,2),r_lo
	lea      (r_hi,r_hi,2),r_hi
	movdqu   8(r_ptr,r_lo,8),%xmm1
	movdqu   8(r_ptr,r_hi,8),%xmm2
	pcmpeqq  %xmm0,%xmm7
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	por      %xmm7,%xmm1
	movq     (r_ptr,r_lo,8),%xmm3
	movhps   (r_ptr,r_hi,8),%xmm3
	pcmpeqq  %xmm0,%xmm3
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	pmovmskb %xmm1,%eax
	HAS_RET
END(has3st1sse4)

FUNC(has4st0sse4)
	HAS_REGS
	movd     r_lo,%xmm0
	pinsrd   $1,r_hi,%xmm0
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	mov      m_bb(r_ptr),r_ptr
	movddup  %xmm0,%xmm0
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm1
	movdqa   (r_ptr,r_hi,1),%xmm2
	pcmpeqq  %xmm0,%xmm1
	movdqa   16(r_ptr,r_lo,1),%xmm3
	pcmpeqq  %xmm0,%xmm2
	pcmpeqq  %xmm0,%xmm3
	por      %xmm2,%xmm1
	pcmpeqq  16(r_ptr,r_hi,1),%xmm0
	por      %xmm3,%xmm1
	por      %xmm0,%xmm1
	pmovmskb %xmm1,%eax
	HAS_RET
END(has4st0sse4)

FUNC(add4st0sse4)
	sub      $16,%esp
	mov      %ecx,save_ecx(%esp)
	movd     %eax,%xmm0
	pinsrd   $1,%edx,%xmm0
	and      m_mask(%ecx),%eax
	and      m_mask(%ecx),%edx
	mov      m_bb(%ecx),%ecx
	movddup  %xmm0,%xmm0
	shl      $5,%eax
	shl      $5,%edx
	movdqa   (%ecx,%eax,1),%xmm1
	movdqa   (%ecx,%edx,1),%xmm2
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	por      %xmm2,%xmm1
	movdqa   16(%ecx,%eax,1),%xmm3
	movdqa   16(%ecx,%edx,1),%xmm4
	pcmpeqq  %xmm0,%xmm3
	pcmpeqq  %xmm0,%xmm4
	por      %xmm4,%xmm1
	por      %xmm3,%xmm1
	pmovmskb %xmm1,%ecx
	test     %ecx,%ecx
	jz       NAME(insert4sse4)
	add      $16,%esp
	xor      %eax,%eax
	ret
END(add4st0sse4)

FUNC(insert4sse4)
	// Strategically bump set->cnt.
	mov      save_ecx(%esp),%ecx
	addl     $1,m_cnt(%ecx)
	mov      m_bb(%ecx),%ecx
	mov      %ebx,save_ebx(%esp)

	// Blank values for b1[*] + b2[*] -> xmm3.
	movd     %eax,%xmm3
	pinsrd   $2,%edx,%xmm3
	pxor     %xmm6,%xmm6
	pcmpeqq  %xmm6,%xmm3

	// Load b1[0] + b2[0], check for a free slot.
	movq     (%ecx,%eax,1),%xmm1
	movhps   (%ecx,%edx,1),%xmm1
	pcmpeqq  %xmm3,%xmm1
	pmovmskb %xmm1,%ebx
	test     %ebx,%ebx
	jnz      2f

	// Load b1[1] + b2[1], check for a free slot.
	movq     8(%ecx,%eax,1),%xmm1
	movhps   8(%ecx,%edx,1),%xmm1
	pcmpeqq  %xmm3,%xmm1
	pmovmskb %xmm1,%ebx
	test     %ebx,%ebx
	jnz      3f

	// Load b1[2] + b2[2], check for a free slot.
	movq     16(%ecx,%eax,1),%xmm1
	movhps   16(%ecx,%edx,1),%xmm1
	pcmpeqq  %xmm3,%xmm1
	pmovmskb %xmm1,%ebx
	test     %ebx,%ebx
	jnz      4f

	// Load b1[3] + b2[3], check for a free slot.
	movq     24(%ecx,%eax,1),%xmm1
	movhps   24(%ecx,%edx,1),%xmm1
	pcmpeqq  %xmm3,%xmm1
	pmovmskb %xmm1,%ebx
	test     %ebx,%ebx
	jnz      5f

	// Nope, preparing for the kickAdd() loop.
	mov      %esi,save_esi(%esp)
	mov      %edi,save_edi(%esp)
	mov      save_ecx(%esp),%ebx
	mov      m_logsize(%ebx),%esi
	shl      $1,%esi
	mov      m_mask(%ebx),%ebx
	jmp      10f

	// Successful justAdd() lands here.
2:	test     %bl,%bl
	cmovz    %edx,%eax
	mov      save_ebx(%esp),%ebx
	add      $16,%esp
	movq     %xmm0,(%ecx,%eax,1)
	mov      $1,%eax
	ret
3:	test     %bl,%bl
	cmovz    %edx,%eax
	mov      save_ebx(%esp),%ebx
	add      $16,%esp
	movq     %xmm0,8(%ecx,%eax,1)
	mov      $1,%eax
	ret
4:	test     %bl,%bl
	cmovz    %edx,%eax
	mov      save_ebx(%esp),%ebx
	add      $16,%esp
	movq     %xmm0,16(%ecx,%eax,1)
	mov      $1,%eax
	ret
5:	test     %bl,%bl
	cmovz    %edx,%eax
	mov      save_ebx(%esp),%ebx
	add      $16,%esp
	movq     %xmm0,24(%ecx,%eax,1)
	mov      $1,%eax
	ret

	// Loop: the element in xmm0 is pushed at eax from the top.
	// The element kicked out from the bottom goes into xmm7.
	.align   32
10:	movq     (%ecx,%eax,1),%xmm7
	movdqu   8(%ecx,%eax,1),%xmm1
	movq     24(%ecx,%eax,1),%xmm2
	movdqa   %xmm1,(%ecx,%eax,1)
	movq     %xmm2,16(%ecx,%eax,1)
	movq     %xmm0,24(%ecx,%eax,1)

	// Find the alternative bucket for xmm7 at edx.
	movd     %xmm7,%edx
	pextrd   $1,%xmm7,%edi
	and      %ebx,%edx
	and      %ebx,%edi
	shl      $5,%edx
	shl      $5,%edi
	cmp      %eax,%edx
	cmovz    %edi,%edx

	// Make the blank value for bb[edx].
	movd     %edx,%xmm1
	pinsrd   $2,%edx,%xmm1
	pcmpeqq  %xmm6,%xmm1

	// Check for empty slots and insert.
	movdqa   16(%ecx,%edx,1),%xmm2
	pcmpeqq  %xmm1,%xmm2
	pcmpeqq  (%ecx,%edx,1),%xmm1
	pmovmskb %xmm2,%eax
	pmovmskb %xmm1,%edi
	shl      $16,%eax
	or       %edi,%eax
	jnz      12f

	// No free slot at edx.
	movdqa   %xmm7,%xmm0
	mov      %edx,%eax

	// Loop control.
	sub      $1,%esi
	jge      10b

	// Broke out of the loop.  Too bad.
	mov      save_ecx(%esp),%ecx
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	add      $16,%esp
	movd     %xmm0,%eax
	pextrd   $1,%xmm0,%edx
	jmp      NAME(insert4tail)

	// Successful kickAdd() lands here.
12:	bsf      %eax,%eax
	add      %eax,%edx
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	mov      save_edi(%esp),%edi
	movq     %xmm7,(%ecx,%edx,1)
	add      $16,%esp
	mov      $1,%eax
	ret
END(insert4sse4)

FUNC(add4st1sse4)
	sub      $16,%esp
	mov      %ecx,save_ecx(%esp)
	movd     %eax,%xmm0
	pinsrd   $1,%edx,%xmm0
	movdqa   m_stash(%ecx),%xmm7
	and      m_mask(%ecx),%eax
	and      m_mask(%ecx),%edx
	mov      m_bb(%ecx),%ecx
	movddup  %xmm0,%xmm0
	shl      $5,%eax
	shl      $5,%edx
	movdqa   (%ecx,%eax,1),%xmm1
	movdqa   (%ecx,%edx,1),%xmm2
	pcmpeqq  %xmm0,%xmm7
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	movdqa   16(%ecx,%eax,1),%xmm3
	movdqa   16(%ecx,%edx,1),%xmm4
	pcmpeqq  %xmm0,%xmm3
	pcmpeqq  %xmm0,%xmm4
	por      %xmm7,%xmm1
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	por      %xmm4,%xmm1
	pmovmskb %xmm1,%ecx
	test     %ecx,%ecx
	jz       NAME(insert4sse4)
	add      $16,%esp
	xor      %eax,%eax
	ret
END(add4st1sse4)

FUNC(has4st1sse4)
	HAS_REGS
	movd     r_lo,%xmm0
	pinsrd   $1,r_hi,%xmm0
	movd     r_lo,%xmm0
	pinsrd   $1,r_hi,%xmm0
	movdqa   m_stash(r_ptr),%xmm7
	and      m_mask(r_ptr),r_lo
	and      m_mask(r_ptr),r_hi
	mov      m_bb(r_ptr),r_ptr
	movddup  %xmm0,%xmm0
	shl      $4,r_lo
	shl      $4,r_hi
	movdqa   (r_ptr,r_lo,1),%xmm1
	movdqa   (r_ptr,r_hi,1),%xmm2
	pcmpeqq  %xmm0,%xmm7
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	movdqa   16(r_ptr,r_lo,1),%xmm3
	movdqa   16(r_ptr,r_hi,1),%xmm4
	pcmpeqq  %xmm0,%xmm3
	pcmpeqq  %xmm0,%xmm4
	por      %xmm7,%xmm1
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	por      %xmm4,%xmm1
	pmovmskb %xmm1,%eax
	HAS_RET
END(has4st1sse4)
