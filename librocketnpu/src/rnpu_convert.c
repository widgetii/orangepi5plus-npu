/*
 * NHWC <-> NPU interleaved format conversion
 * SPDX-License-Identifier: MIT
 */

#include "rnpu_convert.h"

#ifdef __aarch64__
#include <arm_neon.h>
#endif

unsigned rnpu_calc_npu_tensor_size(unsigned w, unsigned h, unsigned c)
{
   unsigned groups = DIV_ROUND_UP(c, FEATURE_ATOMIC_SIZE) * 2;
   return w * h * groups * FEATURE_ATOMIC_SIZE;
}

unsigned rnpu_calc_raw_output_size(unsigned w, unsigned h, unsigned oc,
                                   unsigned tensor_oc)
{
   if (tensor_oc > 0) {
      unsigned num_ops = DIV_ROUND_UP(tensor_oc, oc);
      return w * h * num_ops * 2 * FEATURE_ATOMIC_SIZE;
   }
   unsigned oc1 = DIV_ROUND_UP(oc, FEATURE_ATOMIC_SIZE) * 2;
   return w * h * oc1 * FEATURE_ATOMIC_SIZE;
}

void rnpu_convert_input(uint8_t *npu, const uint8_t *nhwc,
                        unsigned width, unsigned height, unsigned channels,
                        uint8_t zero_point)
{
   if (channels == 1) {
      unsigned n = 0;
      for (unsigned x = 0; x < width; x++)
         for (unsigned y = 0; y < MAX2(height, FEATURE_ATOMIC_SIZE); y++) {
            if (y < height)
               npu[n++] = nhwc[x * height + y];
            else
               npu[n++] = zero_point;
         }
      return;
   }

#ifdef __aarch64__
   if (channels == 3) {
      uint8_t pad_val = (uint8_t)(zero_point - 0x80);
      uint8x16_t pad_vec = vdupq_n_u8(pad_val);
      unsigned n = 0;
      const uint8_t (*in)[height][channels] = (const void *)nhwc;
      for (unsigned x = 0; x < width; x++) {
         for (unsigned y = 0; y < height; y++) {
            vst1q_u8(npu + n, pad_vec);
            npu[n + 0] = in[x][y][0] - 0x80;
            npu[n + 1] = in[x][y][1] - 0x80;
            npu[n + 2] = in[x][y][2] - 0x80;
            n += FEATURE_ATOMIC_SIZE;
         }
      }
      return;
   }
#endif

   unsigned n = 0;
   const uint8_t (*in)[height][channels] = (const void *)nhwc;
   for (unsigned u = 0; u < DIV_ROUND_UP(channels, FEATURE_ATOMIC_SIZE); u++) {
      for (unsigned x = 0; x < width; x++) {
         for (unsigned y = 0; y < height; y++) {
            unsigned base_c = u * FEATURE_ATOMIC_SIZE;
            unsigned real_c = MIN2(FEATURE_ATOMIC_SIZE, channels - base_c);
            for (unsigned c = 0; c < real_c; c++)
               npu[n++] = in[x][y][base_c + c] - 0x80;
            uint8_t pad = zero_point - 0x80;
            for (unsigned c = real_c; c < FEATURE_ATOMIC_SIZE; c++)
               npu[n++] = pad;
         }
      }
   }
}

void rnpu_convert_output(uint8_t *nhwc, const uint8_t *npu,
                         unsigned width, unsigned height, unsigned channels)
{
   unsigned groups = DIV_ROUND_UP(channels, FEATURE_ATOMIC_SIZE);
   uint8_t (*out)[width][channels] = (void *)nhwc;

   if (groups == 1) {
      const uint8_t *src = npu;
      for (unsigned y = 0; y < height; y++) {
         for (unsigned x = 0; x < width; x++) {
#ifdef __aarch64__
            if (channels == FEATURE_ATOMIC_SIZE) {
               uint8x16_t v = vld1q_u8(src);
               v = vaddq_u8(v, vdupq_n_u8(0x80));
               vst1q_u8(&out[y][x][0], v);
            } else {
               for (unsigned c = 0; c < channels; c++)
                  out[y][x][c] = src[c] + 0x80;
            }
#else
            for (unsigned c = 0; c < channels; c++)
               out[y][x][c] = src[c] + 0x80;
#endif
            src += FEATURE_ATOMIC_SIZE;
         }
      }
   } else {
      for (unsigned g = 0; g < groups; g++) {
         unsigned base_c = g * FEATURE_ATOMIC_SIZE;
         unsigned real_c = MIN2(FEATURE_ATOMIC_SIZE, channels - base_c);
         const uint8_t *gb = npu + g * height * width * FEATURE_ATOMIC_SIZE;
         for (unsigned y = 0; y < height; y++)
            for (unsigned x = 0; x < width; x++) {
               const uint8_t *src = gb + (y * width + x) * FEATURE_ATOMIC_SIZE;
               for (unsigned c = 0; c < real_c; c++)
                  out[y][x][base_c + c] = src[c] + 0x80;
            }
      }
   }
}
