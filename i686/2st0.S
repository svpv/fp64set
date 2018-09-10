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
