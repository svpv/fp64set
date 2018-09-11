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

FUNC(has2st0sse4)
	movd     %eax,%xmm0
	pinsrd   $1,%edx,%xmm0
	and      m_mask(%ecx),%eax
	and      m_mask(%ecx),%edx
	mov      m_bb(%ecx),%ecx
	movddup  %xmm0,%xmm0
	shl      $4,%eax
	shl      $4,%edx
	movdqa   (%ecx,%eax,1),%xmm1
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  (%ecx,%edx,1),%xmm0
	por      %xmm1,%xmm0
	pmovmskb %xmm0,%eax
	ret
END(has2st0sse4)

FUNC(add2st0sse4)
	sub      $16,%esp
#define save_ecx 0
#define save_ebx 4
#define save_esi 8
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
	jz       1f
	add      $16,%esp
	xor      %eax,%eax
	ret

	// Strategically bump set->cnt.
1:	mov      save_ecx(%esp),%ecx
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
	and      %ebx,%edx
	shl      $4,%edx
	cmp      %eax,%edx
	jne      11f
	pextrd   $1,%xmm7,%edx
	and      %ebx,%edx
	shl      $4,%edx

	// Make the blank value for bb[edx].
11:	movd     %edx,%xmm1
	pinsrd   $2,%edx,%xmm1
	pcmpeqq  %xmm6,%xmm1

	// Check for empty slots and insert.
	pcmpeqq  (%ecx,%edx,1),%xmm1
	pmovmskb %xmm1,%eax
	test     %al,%al
	jnz      12f
	test     %eax,%eax
	jnz      13f

	// No free slot at edx.
	movdqa   %xmm7,%xmm0
	mov      %edx,%eax

	// Loop control.
	sub      $1,%esi
	jge      10b

	// Broke out of the loop.
	jmp      20f

	// Successful kickAdd() lands here.
13:	add      $8,%edx
12:	movq     %xmm7,(%ecx,%edx,1)
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	add      $16,%esp
	mov      $1,%eax
	ret

	// Too bad.  By the rivers of Babylon we sat and wept.
20:	movddup  %xmm0,%xmm0
	mov      save_ecx(%esp),%ecx
	mov      save_ebx(%esp),%ebx
	mov      save_esi(%esp),%esi
	add      $16,%esp
	movdqa   %xmm0,m_stash(%ecx)
	subl     $1,m_cnt(%ecx)
	movl     $1,m_nstash(%ecx)
	movl     FPTR(add2st1sse4),m_add(%ecx)
	movl     FPTR(has2st1sse4),m_has(%ecx)
	movl     FPTR(del2st1),m_del(%ecx)
	mov      $1,%eax
	ret
END(add2st0sse4)
