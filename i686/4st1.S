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

FUNC(has4st1sse4)
	movd     %eax,%xmm0
	pinsrd   $0x1,%edx,%xmm0
	movdqa   m_stash(%ecx),%xmm7
	and      m_mask(%ecx),%eax
	and      m_mask(%ecx),%edx
	mov      m_bb(%ecx),%ecx
	movddup  %xmm0,%xmm0
	shl      $0x5,%eax
	shl      $0x5,%edx
	movdqa   (%ecx,%eax,1),%xmm1
	movdqa   (%ecx,%edx,1),%xmm2
	pcmpeqq  %xmm0,%xmm7
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	movdqa   0x10(%ecx,%eax,1),%xmm3
	movdqa   0x10(%ecx,%edx,1),%xmm4
	pcmpeqq  %xmm0,%xmm3
	pcmpeqq  %xmm0,%xmm4
	por      %xmm7,%xmm1
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	por      %xmm4,%xmm1
	pmovmskb %xmm1,%eax
	ret
END(has4st1sse4)
