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
	movd     %eax,%xmm0
	pinsrd   $0x1,%edx,%xmm0
	and      m_mask(%ecx),%eax
	and      m_mask(%ecx),%edx
	movdqa   m_stash(%ecx),%xmm7
	movddup  %xmm0,%xmm0
	mov      m_bb(%ecx),%ecx
	lea      (%eax,%eax,2),%eax
	lea      (%edx,%edx,2),%edx
	movdqu   0x8(%ecx,%eax,8),%xmm1
	movdqu   0x8(%ecx,%edx,8),%xmm2
	pcmpeqq  %xmm0,%xmm7
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	por      %xmm7,%xmm1
	movq     (%ecx,%eax,8),%xmm3
	movhps   (%ecx,%edx,8),%xmm3
	pcmpeqq  %xmm0,%xmm3
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	pmovmskb %xmm1,%eax
	ret
END(has3st1sse4)
