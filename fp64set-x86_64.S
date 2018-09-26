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

#define FUNC(name)  \
	.global  fp64set_##name; \
	.hidden  fp64set_##name; \
	.type    fp64set_##name,@function; \
	.align   16; \
	fp64set_##name: \

#define END(name) \
	.size    fp64set_##name,.-fp64set_##name

#define m_mask   40
#define m_bb     48
#define m_stash  0

#if defined(_WIN32) || defined(__CYGWIN__)
#define MSABI    1
#else
#define MSABI    0
#endif

#if MSABI
#define r_fp     %rcx
#define r_ptr    %rdx
#else
#define r_fp     %rdi
#define r_ptr    %rsi
#endif

FUNC(has2st0sse4)
	mov      r_fp,%rax
	movq     r_fp,%xmm0
	shr      $32,%rax
	and      m_mask(r_ptr),r_fp
	and      m_mask(r_ptr),%rax
	mov      m_bb(r_ptr),%r11
	shl      $4,r_fp
	movddup  %xmm0,%xmm0
	movdqa   (%r11,r_fp,1),%xmm1
	pcmpeqq  %xmm0,%xmm1
	shl      $4,%rax
	pcmpeqq  (%r11,%rax,1),%xmm0
	por      %xmm0,%xmm1
	pmovmskb %xmm1,%eax
	retq
END(has2st0sse4)

FUNC(has2st1sse4)
	mov      r_fp,%rax
	movq     r_fp,%xmm0
	shr      $32,%rax
	and      m_mask(r_ptr),r_fp
	and      m_mask(r_ptr),%rax
	mov      m_bb(r_ptr),%r11
	shl      $4,r_fp
	movddup  %xmm0,%xmm0
	movdqa   (%r11,r_fp,1),%xmm1
	pcmpeqq  %xmm0,%xmm1
	movdqa   m_stash(r_ptr),%xmm2
	pcmpeqq  %xmm0,%xmm2
	shl      $4,%rax
	pcmpeqq  (%r11,%rax,1),%xmm0
	por      %xmm2,%xmm1
	por      %xmm0,%xmm1
	pmovmskb %xmm1,%eax
	retq
END(has2st1sse4)

FUNC(has3st0sse4)
	mov      m_mask(%rsi),%rax
	mov      m_bb(%rsi),%rcx
	mov      %rax,%rdx
	and      %rdi,%rdx
	movq     %rdi,%xmm0
	shr      $32,%rdi
	and      %rax,%rdi
	lea      (%rdx,%rdx,2),%rax
	lea      (%rdi,%rdi,2),%rdx
	movdqu   (%rcx,%rax,8),%xmm1
	movdqu   (%rcx,%rdx,8),%xmm2
	pshufd   $0x44,%xmm0,%xmm0
	movq     16(%rcx,%rax,8),%xmm3
	movq     16(%rcx,%rdx,8),%xmm4
	punpcklqdq %xmm3,%xmm4
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	por      %xmm1,%xmm2
	pcmpeqq  %xmm0,%xmm4
	por      %xmm2,%xmm4
	pmovmskb %xmm4,%eax
	retq
END(has3st0sse4)

FUNC(has3st1sse4)
	mov      m_mask(%rsi),%rax
	mov      m_bb(%rsi),%rcx
	mov      %rax,%rdx
	and      %rdi,%rdx
	movq     %rdi,%xmm0
	shr      $32,%rdi
	and      %rax,%rdi
	lea      (%rdx,%rdx,2),%rax
	lea      (%rdi,%rdi,2),%rdx
	movdqu   (%rcx,%rax,8),%xmm1
	movdqu   (%rcx,%rdx,8),%xmm2
	pshufd   $0x44,%xmm0,%xmm0
	movq     16(%rcx,%rax,8),%xmm3
	movq     16(%rcx,%rdx,8),%xmm4
	punpcklqdq %xmm3,%xmm4
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	pcmpeqq  %xmm0,%xmm4
	pcmpeqq  m_stash(%rsi),%xmm0
	por      %xmm1,%xmm2
	por      %xmm2,%xmm0
	por      %xmm4,%xmm0
	pmovmskb %xmm0,%eax
	retq
END(has3st1sse4)

FUNC(has4st0sse4)
	mov      m_mask(%rsi),%rax
	mov      m_bb(%rsi),%rcx
	mov      %rax,%rdx
	and      %rdi,%rdx
	movq     %rdi,%xmm0
	shr      $32,%rdi
	and      %rax,%rdi
	shl      $5,%rdx
	shl      $5,%rdi
	pshufd   $0x44,%xmm0,%xmm0
	movdqa   (%rcx,%rdx,1),%xmm1
	pcmpeqq  %xmm0,%xmm1
	movdqa   (%rcx,%rdi,1),%xmm2
	pcmpeqq  %xmm0,%xmm2
	movdqa   16(%rcx,%rdx,1),%xmm3
	pcmpeqq  %xmm0,%xmm3
	pcmpeqq  16(%rcx,%rdi,1),%xmm0
	por      %xmm1,%xmm2
	por      %xmm3,%xmm0
	por      %xmm2,%xmm0
	pmovmskb %xmm0,%eax
	retq
END(has4st0sse4)

FUNC(has4st1sse4)
	mov      m_mask(%rsi),%rax
	mov      m_bb(%rsi),%rcx
	mov      %rax,%rdx
	and      %rdi,%rdx
	movq     %rdi,%xmm0
	shr      $32,%rdi
	and      %rax,%rdi
	shl      $5,%rdx
	shl      $5,%rdi
	pshufd   $0x44,%xmm0,%xmm0
	movdqa   (%rcx,%rdx,1),%xmm1
	pcmpeqq  %xmm0,%xmm1
	movdqa   (%rcx,%rdi,1),%xmm2
	pcmpeqq  %xmm0,%xmm2
	por      %xmm1,%xmm2
	movdqa   m_stash(%rsi),%xmm1
	pcmpeqq  %xmm0,%xmm1
	movdqa   16(%rcx,%rdx,1),%xmm3
	pcmpeqq  %xmm0,%xmm3
	por      %xmm1,%xmm3
	por      %xmm2,%xmm3
	pcmpeqq  16(%rcx,%rdi,1),%xmm0
	por      %xmm3,%xmm0
	pmovmskb %xmm0,%eax
	retq
END(has4st1sse4)
