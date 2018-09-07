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

FUNC(has2st0sse4)
	movd     %eax,%xmm0
	pinsrd   $0x1,%edx,%xmm0
	and      0x1c(%ecx),%eax
	and      0x1c(%ecx),%edx
	mov      0x20(%ecx),%ecx
	movddup  %xmm0,%xmm0
	shl      $0x4,%eax
	shl      $0x4,%edx
	movdqa   (%ecx,%eax,1),%xmm1
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  (%ecx,%edx,1),%xmm0
	por      %xmm1,%xmm0
	pmovmskb %xmm0,%eax
	ret
END(has2st0sse4)

FUNC(has2st1sse4)
	movd     %eax,%xmm0
	pinsrd   $0x1,%edx,%xmm0
	movdqa   (%ecx),%xmm2
	and      0x1c(%ecx),%eax
	and      0x1c(%ecx),%edx
	mov      0x20(%ecx),%ecx
	movddup  %xmm0,%xmm0
	shl      $0x4,%eax
	shl      $0x4,%edx
	movdqa   (%ecx,%eax,1),%xmm1
	pcmpeqq  %xmm0,%xmm2
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  (%ecx,%edx,1),%xmm0
	por      %xmm2,%xmm0
	por      %xmm1,%xmm0
	pmovmskb %xmm0,%eax
	ret
END(has2st1sse4)

FUNC(has3st0sse4)
	movd     %eax,%xmm0
	pinsrd   $0x1,%edx,%xmm0
	and      0x1c(%ecx),%eax
	and      0x1c(%ecx),%edx
	mov      0x20(%ecx),%ecx
	lea      (%eax,%eax,2),%eax
	lea      (%edx,%edx,2),%edx
	movddup  %xmm0,%xmm0
	movdqu   0x8(%ecx,%eax,8),%xmm1
	movdqu   0x8(%ecx,%edx,8),%xmm2
	movq     (%ecx,%eax,8),%xmm3
	movhps   (%ecx,%edx,8),%xmm3
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	pcmpeqq  %xmm0,%xmm3
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	pmovmskb %xmm1,%eax
	ret
END(has3st0sse4)

FUNC(has3st1sse4)
	movd     %eax,%xmm0
	pinsrd   $0x1,%edx,%xmm0
	movdqa   (%ecx),%xmm4
	and      0x1c(%ecx),%eax
	and      0x1c(%ecx),%edx
	mov      0x20(%ecx),%ecx
	lea      (%eax,%eax,2),%eax
	lea      (%edx,%edx,2),%edx
	movddup  %xmm0,%xmm0
	movdqu   0x8(%ecx,%eax,8),%xmm1
	movdqu   0x8(%ecx,%edx,8),%xmm2
	movq     (%ecx,%eax,8),%xmm3
	movhps   (%ecx,%edx,8),%xmm3
	pcmpeqq  %xmm0,%xmm4
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  %xmm0,%xmm2
	pcmpeqq  %xmm0,%xmm3
	por      %xmm4,%xmm1
	por      %xmm2,%xmm1
	por      %xmm3,%xmm1
	pmovmskb %xmm1,%eax
	ret
END(has3st1sse4)

FUNC(has4st0sse4)
	push     %edi
	push     %esi
	mov      0x1c(%ecx),%esi
	mov      0x20(%ecx),%ecx
	mov      %esi,%edi
	and      %eax,%edi
	and      %edx,%esi
	shl      $0x5,%edi
	shl      $0x5,%esi
	movd     %eax,%xmm0
	pinsrd   $0x1,%edx,%xmm0
	pinsrd   $0x2,%eax,%xmm0
	pinsrd   $0x3,%edx,%xmm0
	movdqa   (%ecx,%edi,1),%xmm1
	pcmpeqq  %xmm0,%xmm1
	movdqa   (%ecx,%esi,1),%xmm2
	pcmpeqq  %xmm0,%xmm2
	por      %xmm1,%xmm2
	movdqa   0x10(%ecx,%edi,1),%xmm1
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  0x10(%ecx,%esi,1),%xmm0
	por      %xmm1,%xmm0
	pmovmskb %xmm2,%ecx
	pmovmskb %xmm0,%eax
	or       %ecx,%eax
	pop      %esi
	pop      %edi
	ret
END(has4st0sse4)

FUNC(has4st1sse4)
	push     %ebx
	push     %edi
	push     %esi
	mov      0x1c(%ecx),%esi
	mov      0x20(%ecx),%edi
	mov      %esi,%ebx
	and      %eax,%ebx
	and      %edx,%esi
	shl      $0x5,%ebx
	shl      $0x5,%esi
	movd     %eax,%xmm0
	pinsrd   $0x1,%edx,%xmm0
	pinsrd   $0x2,%eax,%xmm0
	pinsrd   $0x3,%edx,%xmm0
	movdqa   (%edi,%ebx,1),%xmm1
	pcmpeqq  %xmm0,%xmm1
	movdqa   (%edi,%esi,1),%xmm2
	pcmpeqq  %xmm0,%xmm2
	por      %xmm1,%xmm2
	movdqa   (%ecx),%xmm1
	pcmpeqq  %xmm0,%xmm1
	movdqa   0x10(%edi,%ebx,1),%xmm3
	pcmpeqq  %xmm0,%xmm3
	pcmpeqq  0x10(%edi,%esi,1),%xmm0
	por      %xmm3,%xmm0
	pmovmskb %xmm1,%ecx
	pmovmskb %xmm2,%edx
	pmovmskb %xmm0,%eax
	or       %ecx,%eax
	or       %edx,%eax
	pop      %esi
	pop      %edi
	pop      %ebx
	ret
END(has4st1sse4)
