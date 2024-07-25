# Data-Compression
## Task1: Huffman and QMCoder
### Huffman
- Implement lossless coding using Huffman coding.
- Use both the original data byte and Differential Pulse Code Modulation (DPCM) as the data samples.
- Implement a Huffman coder to compress the image files.
- Transmit the codewords in the bitstream.
- Report the compressed file sizes (excluding the size of transmitted codewords).
- Compare the results with the first-order entropy.
- Note: Ignore potential issues at the end of the decoding process assuming the decoder knows the image resolution.
- Run code: check 'Task1_Huffman and QMCoder/README.md'

### QMCoder
- Implement the QM encoder for binary arithmetic coding.
- Use two parameters in the coder:
  - `A`: Records the size of the tag interval with a value between 0.75 (0x8000) and 1.5 (0x10000).
  - `C`: The lower bound that stores the encoded bits.
- Refer to the provided pseudo-code of a binary arithmetic encoder for implementation.
- Run code: check 'Task1_Huffman and QMCoder/README.md'

## Task2: JPEG-like Compression
## Task2: JPEG-like Compression

### JPEG-like Compression
- Implement still image transform coding similar to a prototype JPEG.
- Use any programming language of your choice.
- Test images:
  - Gray-level: `lena.raw`, `baboon.raw`
  - Color: `lenaRGB.raw`, `baboonRGB.raw`
- Download Irfanview from [here](http://www.irfanview.com) for viewing images.
- Utilize the TMN version of 8x8 DCT, optimized for H.263 video coding.
  - Find or build the routine for 8x8 DCT.
  - Apply quantization and coding to compress the images.
  - Adjust the Quality Factor (QF) to determine the quantization table.
- Use the differential coding tables of luminance DC and AC provided in the class notes.
- Ensure the image is recoverable to `.raw` format for viewing.
- Calculate PSNR of the original and compressed image for QF values: 90, 80, 50, 20, 10, and 5.
- Run code: check 'Task2_JPEG-like Compression/README.md'
