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
	push     %edi
	push     %esi
	mov      0x1c(%ecx),%esi
	mov      0x20(%ecx),%ecx
	mov      %esi,%edi
	and      %eax,%edi
	and      %edx,%esi
	shl      $0x4,%edi
	shl      $0x4,%esi
	movd     %eax,%xmm0
	pinsrd   $0x1,%edx,%xmm0
	pinsrd   $0x2,%eax,%xmm0
	pinsrd   $0x3,%edx,%xmm0
	movdqa   (%ecx,%edi,1),%xmm1
	pcmpeqq  %xmm0,%xmm1
	pcmpeqq  (%ecx,%esi,1),%xmm0
	pmovmskb %xmm1,%ecx
	pmovmskb %xmm0,%eax
	or       %ecx,%eax
	pop      %esi
	pop      %edi
	ret
END(has2st0sse4)

FUNC(has2st1sse4)
	push     %ebx
	push     %edi
	push     %esi
	mov      0x1c(%ecx),%esi
	mov      0x20(%ecx),%edi
	mov      %esi,%ebx
	and      %eax,%ebx
	and      %edx,%esi
	shl      $0x4,%ebx
	shl      $0x4,%esi
	movd     %eax,%xmm0
	pinsrd   $0x1,%edx,%xmm0
	pinsrd   $0x2,%eax,%xmm0
	pinsrd   $0x3,%edx,%xmm0
	movdqa   (%edi,%ebx,1),%xmm1
	pcmpeqq  %xmm0,%xmm1
	movdqa   (%edi,%esi,1),%xmm2
	pcmpeqq  %xmm0,%xmm2
	pcmpeqq  (%ecx),%xmm0
	pmovmskb %xmm1,%ecx
	pmovmskb %xmm2,%edx
	pmovmskb %xmm0,%eax
	or       %ecx,%eax
	or       %edx,%eax
	pop      %esi
	pop      %edi
	pop      %ebx
	ret
END(has2st1sse4)

FUNC(has3st0sse4)
	push     %edi
	push     %esi
	mov      0x1c(%ecx),%esi
	mov      0x20(%ecx),%ecx
	mov      %esi,%edi
	and      %eax,%edi
	and      %edx,%esi
	lea      (%edi,%edi,2),%edi
	lea      (%esi,%esi,2),%esi
	movdqu   (%ecx,%edi,8),%xmm0
	movdqu   (%ecx,%esi,8),%xmm1
	movd     %eax,%xmm2
	pinsrd   $0x1,%edx,%xmm2
	pinsrd   $0x2,%eax,%xmm2
	pinsrd   $0x3,%edx,%xmm2
	movd     0x10(%ecx,%esi,8),%xmm3
	pinsrd   $0x1,0x14(%ecx,%esi,8),%xmm3
	pinsrd   $0x2,0x10(%ecx,%edi,8),%xmm3
	pinsrd   $0x3,0x14(%ecx,%edi,8),%xmm3
	pcmpeqq  %xmm2,%xmm0
	pcmpeqq  %xmm2,%xmm1
	pcmpeqq  %xmm2,%xmm3
	pmovmskb %xmm0,%ecx
	pmovmskb %xmm1,%edx
	pmovmskb %xmm3,%eax
	or       %ecx,%eax
	or       %edx,%eax
	pop      %esi
	pop      %edi
	ret
END(has3st0sse4)

FUNC(has3st1sse4)
	push     %ebx
	push     %edi
	push     %esi
	mov      0x1c(%ecx),%esi
	mov      0x20(%ecx),%edi
	mov      %esi,%ebx
	and      %eax,%ebx
	and      %edx,%esi
	lea      (%ebx,%ebx,2),%ebx
	lea      (%esi,%esi,2),%esi
	movdqu   (%edi,%ebx,8),%xmm0
	movdqu   (%edi,%esi,8),%xmm1
	movd     %eax,%xmm2
	pinsrd   $0x1,%edx,%xmm2
	pinsrd   $0x2,%eax,%xmm2
	pinsrd   $0x3,%edx,%xmm2
	movd     0x10(%edi,%esi,8),%xmm3
	pinsrd   $0x1,0x14(%edi,%esi,8),%xmm3
	pinsrd   $0x2,0x10(%edi,%ebx,8),%xmm3
	pinsrd   $0x3,0x14(%edi,%ebx,8),%xmm3
	pcmpeqq  %xmm2,%xmm0
	pcmpeqq  %xmm2,%xmm1
	por      %xmm0,%xmm1
	pcmpeqq  %xmm2,%xmm3
	pcmpeqq  (%ecx),%xmm2
	por      %xmm3,%xmm2
	pmovmskb %xmm1,%ecx
	pmovmskb %xmm2,%eax
	or       %ecx,%eax
	pop      %esi
	pop      %edi
	pop      %ebx
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
