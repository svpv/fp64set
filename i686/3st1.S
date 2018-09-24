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

FUNC(has3st1sse4)
	HAS_REGS
	movd     reg_lo,%xmm0
	pinsrd   $1,reg_hi,%xmm0
	and      m_mask(reg_ptr),reg_lo
	and      m_mask(reg_ptr),reg_hi
	movdqa   m_stash(reg_ptr),%xmm7
	movddup  %xmm0,%xmm0
	mov      m_bb(reg_ptr),reg_ptr
	lea      (reg_lo,reg_lo,2),reg_lo
	lea      (reg_hi,reg_hi,2),reg_hi
	movdqu   8(reg_ptr,reg_lo,8),%xmm1
	movdqu   8(reg_ptr,reg_hi,8),%xmm2
	pcmpeqq  %xmm0,%xmm7
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	por      %xmm7,%xmm1
	movq     (reg_ptr,reg_lo,8),%xmm3
	movhps   (reg_ptr,reg_hi,8),%xmm3
	pcmpeqq  %xmm0,%xmm3
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	pmovmskb %xmm1,%eax
	HAS_RET
END(has3st1sse4)

FUNC(add3st1sse4)
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
