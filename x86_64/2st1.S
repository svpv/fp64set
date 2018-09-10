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

FUNC(has2st1sse4)
	mov      m_mask(%rsi),%rax
	mov      m_bb(%rsi),%rcx
	mov      %rax,%rdx
	and      %rdi,%rdx
	movq     %rdi,%xmm0
	shr      $32,%rdi
	and      %rax,%rdi
	shl      $4,%rdx
	shl      $4,%rdi
	pshufd   $0x44,%xmm0,%xmm0
	movdqa   (%rcx,%rdx,1),%xmm1
	pcmpeqq  %xmm0,%xmm1
	movdqa   (%rcx,%rdi,1),%xmm2
	pcmpeqq  %xmm0,%xmm2
	pcmpeqq  m_stash(%rsi),%xmm0
	pmovmskb %xmm1,%eax
	pmovmskb %xmm2,%ecx
	or       %eax,%ecx
	pmovmskb %xmm0,%eax
	or       %ecx,%eax
	retq
END(has2st1sse4)
