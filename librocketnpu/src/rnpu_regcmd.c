/*
 * Register command generation — extracted from Mesa rkt_regcmd.c
 * Generates packed 64-bit register write commands for NPU's PC DMA engine.
 * SPDX-License-Identifier: MIT
 */

#include <stdio.h>
#include "rnpu_regcmd.h"
#include "rnpu_registers.h"
#include "rnpu_drm.h"

/* NVDLA-style round-half-away-from-zero for positive, round-toward-zero
 * for negative exact halves.  Matches QEMU rockchip-npu.c:277. */
static inline int64_t nvdla_shift_right_round64(int64_t value, unsigned shift)
{
   if (shift == 0)
      return value;
   int64_t sign = (value < 0) ? 1 : 0;
   uint64_t guide = (value >> (shift - 1)) & 1;
   uint64_t sticky = (shift > 1)
       ? ((value & ((1ULL << (shift - 1)) - 1)) != 0) : 0;
   int64_t round_up = guide & ((!sign) | sticky);
   return (value >> shift) + round_up;
}

static void emit_raw(uint64_t **p, uint32_t target, uint32_t reg, uint32_t value)
{
   uint64_t v = ((uint64_t)target << 48) | ((uint64_t)value << 16) | (uint64_t)reg;
   *(*p)++ = v;
}

static void emit(uint64_t **p, uint32_t reg, uint32_t value)
{
   emit_raw(p, rkt_get_target(reg) + 0x1, reg, value);
}

#define EMIT(reg, val) emit(&dst, reg, val)

static unsigned fill_standard_regcmd(const struct rnpu_model *model,
                                     const struct rnpu_operation *op,
                                     uint64_t *out, unsigned task_num)
{
   uint64_t *dst = out;
   const struct rnpu_split_task *task = &op->tasks[task_num];
   unsigned num_tasks = op->task_count;
   unsigned ozp = task->output_zero_point;
   unsigned wzp = task->weights_zero_point;
   unsigned offset = ozp - 0x80;

   uint64_t act_base = model->activation_bo.dma_addr;
   uint64_t wt_base = model->weight_bo.dma_addr;
   uint64_t bias_base = model->bias_bo.dma_addr;

   uint32_t con0 = CNA_CBUF_CON0_WEIGHT_BANK(task->weights_banks) |
                   CNA_CBUF_CON0_DATA_BANK(task->input_banks);
   if (task_num > 0 && op->reuse_weights_cbuf)
      con0 |= CNA_CBUF_CON0_WEIGHT_REUSE(1);

   EMIT(REG_CNA_CBUF_CON0, con0);
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   EMIT(REG_CNA_DCOMP_CTRL, 0);

   uint32_t con1 = 0;
   if (task->input_channels_real == 1)
      con1 |= CNA_CONV_CON1_NONALIGN_DMA(1) | CNA_CONV_CON1_GROUP_LINE_OFF(1) |
              CNA_CONV_CON1_ARGB_IN(8);
   if (op->depthwise)
      con1 |= CNA_CONV_CON1_CONV_MODE(3);

   EMIT(REG_CNA_CONV_CON1, con1);
   EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) |
                            DPU_S_POINTER_EXECUTER_PP_EN(1) |
                            DPU_S_POINTER_POINTER_PP_EN(1));
   EMIT(REG_DPU_RDMA_RDMA_S_POINTER,
        DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) |
        DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) |
        DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1));
   EMIT(REG_CNA_CONV_CON1, con1);
   EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_FEATURE_GRAINS(50 + task->stride_y + 1));
   EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_X_STRIDE(task->stride_x) |
                            CNA_CONV_CON3_CONV_Y_STRIDE(task->stride_y));
   EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(task->input_width) |
                             CNA_DATA_SIZE0_DATAIN_HEIGHT(task->input_height));
   EMIT(REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(task->input_channels_real - 1) |
                             CNA_DATA_SIZE1_DATAIN_CHANNEL(task->input_channels));
   EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(task->output_width));
   EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(task->atomic_count));
   EMIT(REG_CNA_WEIGHT_SIZE0, task->weights_width * task->weights_height *
                               task->input_channels * task->weights_kernels);
   EMIT(REG_CNA_WEIGHT_SIZE1, task->weights_width * task->weights_height *
                               task->input_channels);
   EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(task->weights_width) |
                               CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(task->weights_height) |
                               CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(task->weights_kernels));
   EMIT(REG_CNA_CBUF_CON0, con0);
   EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(task->input_data_entries));

   if (task->input_channels_real == 1) {
      unsigned truncate = 14, scale = 16384, cvt_off = 65408;
      if (op->addition_input || op->add_tensor != -1) { truncate = 15; scale = 32388; }
      EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_CVT_TRUNCATE_3(truncate) |
                              CNA_CVT_CON0_CVT_TRUNCATE_2(truncate) |
                              CNA_CVT_CON0_CVT_TRUNCATE_1(truncate) |
                              CNA_CVT_CON0_CVT_TRUNCATE_0(truncate));
      EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(scale) | CNA_CVT_CON1_CVT_OFFSET0(cvt_off));
      EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(scale) | CNA_CVT_CON2_CVT_OFFSET1(cvt_off));
      EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(scale) | CNA_CVT_CON3_CVT_OFFSET2(cvt_off));
      EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(scale) | CNA_CVT_CON4_CVT_OFFSET3(cvt_off));
   } else {
      EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_DATA_SIGN(1) | CNA_CVT_CON0_CVT_TYPE(1) |
                              CNA_CVT_CON0_CVT_BYPASS(1));
      EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
      EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
      EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
      EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
   }

   EMIT(REG_CNA_FC_CON0, 0);
   EMIT(REG_CNA_FC_CON1, 0);
   EMIT(REG_CNA_PAD_CON0, CNA_PAD_CON0_PAD_LEFT(task->pad_left) |
                           CNA_PAD_CON0_PAD_TOP(task->pad_top));

   /* Input activation DMA address */
   uint32_t in_addr = (uint32_t)(act_base + model->tensors[op->input_tensor].offset +
                                 task->input_offset);
   EMIT(REG_CNA_FEATURE_DATA_ADDR, in_addr);
   EMIT(REG_CNA_FC_CON2, 0);
   EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) |
                           CNA_DMA_CON0_DATA_BURST_LEN(15));
   EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(task->input_line_stride));
   EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(task->input_surface_stride));
   EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(op->input_width) |
                                CNA_FC_DATA_SIZE0_DMA_HEIGHT(task->input_height));
   EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(task->input_channels));
   EMIT(REG_CNA_DCOMP_CTRL, 0);
   EMIT(REG_CNA_DCOMP_REGNUM, 0);

   /* Weight DMA address */
   EMIT(REG_CNA_DCOMP_ADDR0, (uint32_t)(wt_base + op->weight_offset));
   EMIT(REG_CNA_DCOMP_AMOUNT0, 0); EMIT(REG_CNA_DCOMP_AMOUNT1, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT2, 0); EMIT(REG_CNA_DCOMP_AMOUNT3, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT4, 0); EMIT(REG_CNA_DCOMP_AMOUNT5, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT6, 0); EMIT(REG_CNA_DCOMP_AMOUNT7, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT8, 0); EMIT(REG_CNA_DCOMP_AMOUNT9, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT10, 0); EMIT(REG_CNA_DCOMP_AMOUNT11, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT12, 0); EMIT(REG_CNA_DCOMP_AMOUNT13, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT14, 0); EMIT(REG_CNA_DCOMP_AMOUNT15, 0);

   EMIT(REG_CNA_CVT_CON5, task->input_channels_real == 1 ? 65535 : 0);

   int32_t pad_con1;
   if (task->weights_width >= 3 && task->input_zero_point == 0x0)
      pad_con1 = 0xffff8080;
   else
      pad_con1 = task->input_zero_point - 0x80;
   if (op->addition_input || op->add_tensor != -1)
      pad_con1 = 0xffffff80;
   if (op->depthwise && task->input_zero_point == 0x8b)
      pad_con1 = 0x0b0b;
   EMIT(REG_CNA_PAD_CON1, pad_con1);

   uint32_t misc_cfg = CORE_MISC_CFG_QD_EN(1);
   if (op->depthwise) misc_cfg |= CORE_MISC_CFG_DW_EN(1);
   EMIT(REG_CORE_MISC_CFG, misc_cfg);
   EMIT(REG_CORE_DATAOUT_SIZE_0,
        CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(task->output_height - 1) |
        CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(task->output_width - 1));
   EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(task->output_channels - 1));
   EMIT(REG_CORE_CLIP_TRUNCATE, CORE_CLIP_TRUNCATE_CLIP_TRUNCATE(op->truncate_bits));
   emit_raw(&dst, CORE | 0x1, 0x3030, 0);

   uint32_t fmc = DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2);
   if (op->depthwise) fmc |= DPU_FEATURE_MODE_CFG_CONV_MODE(3);
   EMIT(REG_DPU_FEATURE_MODE_CFG, fmc);
   EMIT(REG_DPU_DATA_FORMAT, 0);
   EMIT(REG_DPU_OFFSET_PEND, 0);

   /* Output DMA address */
   uint32_t out_addr = (uint32_t)(act_base + model->tensors[op->output_tensor].offset +
                                  task->output_offset + op->per_channel_group_offset);
   EMIT(REG_DPU_DST_BASE_ADDR, out_addr);
   EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(task->output_surface_stride));
   EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(task->output_width - 1));
   EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(task->output_height - 1));
   EMIT(REG_DPU_DATA_CUBE_NOTCH_ADDR, 0);
   EMIT(REG_DPU_DATA_CUBE_CHANNEL,
        DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(task->output_channels_real - 1) |
        DPU_DATA_CUBE_CHANNEL_CHANNEL(task->output_channels - 1));
   EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_ALU_ALGO(2) | DPU_BS_CFG_BS_ALU_SRC(1) |
                         DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1));
   EMIT(REG_DPU_BS_ALU_CFG, 0);
   EMIT(REG_DPU_BS_MUL_CFG, 0);
   EMIT(REG_DPU_BS_RELUX_CMP_VALUE, 0);

   if (op->depthwise) {
      EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(3) |
                               DPU_BS_OW_CFG_SIZE_E_1(3) |
                               DPU_BS_OW_CFG_SIZE_E_0(3));
   } else {
      EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(1) |
                               DPU_BS_OW_CFG_SIZE_E_1(1) |
                               DPU_BS_OW_CFG_SIZE_E_0(1));
   }
   EMIT(REG_DPU_BS_OW_OP, 0);
   EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(task->output_channels - 1));
   EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(task->output_height - 1) |
                              DPU_WDMA_SIZE_1_WIDTH_WDMA(task->output_width - 1));
   EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) |
                         DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));
   EMIT(REG_DPU_BN_ALU_CFG, 0);
   EMIT(REG_DPU_BN_MUL_CFG, 0);
   EMIT(REG_DPU_BN_RELUX_CMP_VALUE, 0);

   if (op->add_tensor != -1) {
      EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_CVT_TYPE(1) | DPU_EW_CFG_EW_DATA_MODE(1) |
                            DPU_EW_CFG_EDATA_SIZE(1) | DPU_EW_CFG_EW_ALU_ALGO(2) |
                            DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) |
                            DPU_EW_CFG_EW_OP_SRC(1));
      EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, op->addition_offset);

      /* Addition scale — hardcoded LUT from Mesa (MBv1-specific) */
      float conv_scale = (task->input_scale * task->weights_scale) / task->output_scale;
      uint32_t scale_bits = fui(conv_scale);
      unsigned shift = 127 + 31 - 32 - (scale_bits >> 23) + 16;
      if (op->truncate_bits > 0) shift--;
      unsigned scale = ((scale_bits >> 9) & 0x7fff) + 1;
      if (scale < 1 << 14) scale |= 1 << 14;

      /* Simplified: use computed scale for ADD. For full MBv1 compat,
       * the hardcoded table from Mesa would go here. */
      EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
      EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0);
      EMIT(REG_DPU_OUT_CVT_OFFSET, offset);
      EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(scale));
      EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(shift - 1));
   } else {
      EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) |
                            DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) |
                            DPU_EW_CFG_EW_BYPASS(1));
      EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, 0);
      EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
      EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0);
      EMIT(REG_DPU_OUT_CVT_OFFSET, offset);

      float conv_scale = (task->input_scale * task->weights_scale) / task->output_scale;
      uint32_t scale_bits = fui(conv_scale);
      unsigned shift = 127 + 31 - 32 - (scale_bits >> 23) + 16;
      if (op->truncate_bits > 0) shift--;
      unsigned scale = ((scale_bits >> 9) & 0x7fff) + 1;
      if (scale < 1 << 14) scale |= 1 << 14;

      EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(scale));
      EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(shift - 1));
   }

   EMIT(REG_DPU_EW_OP_VALUE_0, 0); EMIT(REG_DPU_EW_OP_VALUE_1, 0);
   EMIT(REG_DPU_EW_OP_VALUE_2, 0); EMIT(REG_DPU_EW_OP_VALUE_3, 0);
   EMIT(REG_DPU_EW_OP_VALUE_4, 0); EMIT(REG_DPU_EW_OP_VALUE_5, 0);
   EMIT(REG_DPU_EW_OP_VALUE_6, 0); EMIT(REG_DPU_EW_OP_VALUE_7, 0);
   EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(task->surfaces_per_row));
   emit_raw(&dst, DPU | 0x1, 0x40c4, 0);
   EMIT(REG_DPU_LUT_ACCESS_CFG, 0); EMIT(REG_DPU_LUT_ACCESS_DATA, 0);
   EMIT(REG_DPU_LUT_CFG, 0); EMIT(REG_DPU_LUT_INFO, 0);
   EMIT(REG_DPU_LUT_LE_START, 0); EMIT(REG_DPU_LUT_LE_END, 0);
   EMIT(REG_DPU_LUT_LO_START, 0); EMIT(REG_DPU_LUT_LO_END, 0);
   EMIT(REG_DPU_LUT_LE_SLOPE_SCALE, 0); EMIT(REG_DPU_LUT_LE_SLOPE_SHIFT, 0);
   EMIT(REG_DPU_LUT_LO_SLOPE_SCALE, 0); EMIT(REG_DPU_LUT_LO_SLOPE_SHIFT, 0);

   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(task->output_width - 1));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(task->output_height - 1));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(task->output_channels - 1));

   if (op->add_tensor != -1) {
      uint32_t add_addr = (uint32_t)(act_base + model->tensors[op->add_tensor].offset +
                                     task->output_offset);
      EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, add_addr);
   } else {
      EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0);
   }

   EMIT(REG_DPU_RDMA_RDMA_BRDMA_CFG, DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE(1));

   /* Bias DMA address */
   EMIT(REG_DPU_RDMA_RDMA_BS_BASE_ADDR, (uint32_t)(bias_base + op->bias_offset));
   EMIT(REG_DPU_RDMA_RDMA_NRDMA_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_BN_BASE_ADDR, 0);

   unsigned ew_stride = MAX2(op->output_width * op->output_height, 12);

   if (op->add_tensor != -1) {
      EMIT(REG_DPU_RDMA_RDMA_ERDMA_CFG,
           DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE(1) |
           DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE(1));
      uint32_t ew_base = op->output_width * op->output_height * ATOMIC_K_SIZE;
      uint32_t ew_addr = (uint32_t)(act_base + model->tensors[op->add_tensor].offset +
                                    task->output_offset + ew_base);
      EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, ew_addr);
      EMIT(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
           DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE(ew_stride));
   } else {
      EMIT(REG_DPU_RDMA_RDMA_ERDMA_CFG, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(1));
      EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0);
      EMIT(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 0);
   }

   uint32_t rfmc = DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15);
   if (op->add_tensor != -1)
      rfmc |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE(5);
   else
      rfmc |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(1);
   if (op->depthwise)
      rfmc |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE(3);

   EMIT(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rfmc);
   EMIT(REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0);

   unsigned sn = ew_stride + task->output_width * (op->output_height - task->output_height);
   if (op->input_width == 3) sn = 15;
   if (op->add_tensor != -1) {
      EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH, DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR(sn));
   } else {
      EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH, 0);
   }

   EMIT(REG_DPU_RDMA_RDMA_PAD_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_WEIGHT,
        DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(1) |
        DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(1));

   if (op->add_tensor != -1) {
      EMIT(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH(sn));
   } else {
      EMIT(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, 0);
   }

   /* Chain pointer: always emit a proper PC_BASE_ADDRESS(0) so cross-op
    * chain patching (patch_chain) can OR in the next task's address.
    * PC_REGISTER_AMOUNTS(0) ensures the PC won't follow a null chain. */
   EMIT(REG_PC_BASE_ADDRESS, 0);
   EMIT(REG_PC_REGISTER_AMOUNTS, 0);
   *dst++ = 0x0041000000000000ULL;
   emit_raw(&dst, 0x81, REG_PC_OPERATION_ENABLE,
            PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1));

   return (unsigned)(dst - out);
}

static unsigned fill_per_channel_regcmd(const struct rnpu_model *model,
                                        const struct rnpu_operation *op,
                                        uint64_t *out, unsigned task_num)
{
   uint64_t *dst = out;
   const struct rnpu_split_task *task = &op->tasks[task_num];
   unsigned num_tasks = op->task_count;
   unsigned ozp = task->output_zero_point;
   unsigned wzp = task->weights_zero_point;
   uint64_t act_base = model->activation_bo.dma_addr;
   uint64_t wt_base = model->weight_bo.dma_addr;
   uint64_t bias_base = model->bias_bo.dma_addr;

   uint32_t con0 = CNA_CBUF_CON0_WEIGHT_BANK(task->weights_banks) |
                   CNA_CBUF_CON0_DATA_BANK(task->input_banks);
   EMIT(REG_CNA_CBUF_CON0, con0);
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   EMIT(REG_CNA_DCOMP_CTRL, 0);
   EMIT(REG_CNA_CONV_CON1, 0);
   EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) |
                            DPU_S_POINTER_EXECUTER_PP_EN(1) |
                            DPU_S_POINTER_POINTER_PP_EN(1));
   EMIT(REG_DPU_RDMA_RDMA_S_POINTER,
        DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) |
        DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) |
        DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1));
   EMIT(REG_CNA_CONV_CON1, 0);
   EMIT(REG_CNA_CONV_CON2, 0);
   EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_X_STRIDE(task->stride_x) |
                            CNA_CONV_CON3_CONV_Y_STRIDE(task->stride_y));
   EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(task->input_width) |
                             CNA_DATA_SIZE0_DATAIN_HEIGHT(task->input_height));
   EMIT(REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(task->input_channels_real - 1) |
                             CNA_DATA_SIZE1_DATAIN_CHANNEL(task->input_channels));
   EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(task->output_width));
   EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(task->atomic_count));
   EMIT(REG_CNA_WEIGHT_SIZE0, task->weights_width * task->weights_height *
                               task->input_channels * task->weights_kernels);
   EMIT(REG_CNA_WEIGHT_SIZE1, task->weights_width * task->weights_height *
                               task->input_channels);
   EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(task->weights_width) |
                               CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(task->weights_height) |
                               CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(task->weights_kernels));
   EMIT(REG_CNA_CBUF_CON0, con0);
   EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(task->input_data_entries));
   EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_DATA_SIGN(1) | CNA_CVT_CON0_CVT_TYPE(1) |
                           CNA_CVT_CON0_CVT_BYPASS(1));
   EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
   EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
   EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
   EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
   EMIT(REG_CNA_FC_CON0, 0);
   EMIT(REG_CNA_FC_CON1, 0);
   EMIT(REG_CNA_PAD_CON0, CNA_PAD_CON0_PAD_LEFT(task->pad_left) |
                           CNA_PAD_CON0_PAD_TOP(task->pad_top));
   EMIT(REG_CNA_FEATURE_DATA_ADDR,
        (uint32_t)(act_base + model->tensors[op->input_tensor].offset + task->input_offset));
   EMIT(REG_CNA_FC_CON2, 0);
   EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) | CNA_DMA_CON0_DATA_BURST_LEN(15));
   EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(task->input_line_stride));
   EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(task->input_surface_stride));
   EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(op->input_width) |
                                CNA_FC_DATA_SIZE0_DMA_HEIGHT(task->input_height));
   EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(task->input_channels));
   EMIT(REG_CNA_DCOMP_CTRL, 0);
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   EMIT(REG_CNA_DCOMP_ADDR0, (uint32_t)(wt_base + op->weight_offset));
   EMIT(REG_CNA_DCOMP_AMOUNT0, 0); EMIT(REG_CNA_DCOMP_AMOUNT1, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT2, 0); EMIT(REG_CNA_DCOMP_AMOUNT3, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT4, 0); EMIT(REG_CNA_DCOMP_AMOUNT5, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT6, 0); EMIT(REG_CNA_DCOMP_AMOUNT7, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT8, 0); EMIT(REG_CNA_DCOMP_AMOUNT9, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT10, 0); EMIT(REG_CNA_DCOMP_AMOUNT11, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT12, 0); EMIT(REG_CNA_DCOMP_AMOUNT13, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT14, 0); EMIT(REG_CNA_DCOMP_AMOUNT15, 0);
   EMIT(REG_CNA_CVT_CON5, 0);
   EMIT(REG_CNA_PAD_CON1, task->input_zero_point - 0x80);

   EMIT(REG_CORE_MISC_CFG, CORE_MISC_CFG_QD_EN(1));
   EMIT(REG_CORE_DATAOUT_SIZE_0,
        CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(task->output_height - 1) |
        CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(task->output_width - 1));
   EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(task->output_channels - 1));
   EMIT(REG_CORE_CLIP_TRUNCATE, 0);
   emit_raw(&dst, CORE | 0x1, 0x3030, 0);

   EMIT(REG_DPU_FEATURE_MODE_CFG, DPU_FEATURE_MODE_CFG_BURST_LEN(15) |
                                   DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2));
   EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_BS_MUL_SHIFT_VALUE_NEG(7));
   EMIT(REG_DPU_OFFSET_PEND, 0);
   EMIT(REG_DPU_DST_BASE_ADDR,
        (uint32_t)(act_base + model->tensors[op->output_tensor].offset +
                   task->output_offset + op->per_channel_group_offset));
   EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(task->output_surface_stride));
   EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(task->output_width - 1));
   EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(task->output_height - 1));
   EMIT(REG_DPU_DATA_CUBE_NOTCH_ADDR, 0);
   EMIT(REG_DPU_DATA_CUBE_CHANNEL, 0);
   EMIT(REG_DPU_BS_CFG, 0x13f);
   EMIT(REG_DPU_BS_ALU_CFG, (uint32_t)op->per_channel_bias);
   EMIT(REG_DPU_BS_MUL_CFG, 0);
   EMIT(REG_DPU_BS_RELUX_CMP_VALUE, 0x001f001f);
   EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(1) | DPU_BS_OW_CFG_SIZE_E_1(1) |
                            DPU_BS_OW_CFG_SIZE_E_0(1));
   EMIT(REG_DPU_BS_OW_OP, 0);
   EMIT(REG_DPU_WDMA_SIZE_0, 0);
   EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(task->output_height - 1) |
                              DPU_WDMA_SIZE_1_WIDTH_WDMA(task->output_width - 1));
   EMIT(REG_DPU_BN_CFG, 0x12);
   EMIT(REG_DPU_BN_ALU_CFG, 0);
   EMIT(REG_DPU_BN_MUL_CFG, 0);
   EMIT(REG_DPU_BN_RELUX_CMP_VALUE, 0);
   EMIT(REG_DPU_EW_CFG, 0x383);
   EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, 0);
   EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
   EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0);

   float cs = (task->input_scale * task->weights_scale) / task->output_scale;
   uint32_t sb = fui(cs);
   unsigned shift = 127 + 31 - 32 - (sb >> 23) + 16;
   unsigned scale = ((sb >> 9) & 0x7fff) + 1;
   if (scale < 1 << 14) scale |= 1 << 14;

   /* BS bypassed — fold bias into OUT_CVT_OFFSET (inexact due to ReLU ordering) */
   int32_t base_ofs = (int32_t)(ozp - 0x80);
   int64_t bias_x_scale = (int64_t)op->per_channel_bias * (int64_t)scale;
   int32_t bias_out = (int32_t)nvdla_shift_right_round64(bias_x_scale, shift - 1);
   EMIT(REG_DPU_OUT_CVT_OFFSET, (uint32_t)(base_ofs + bias_out));
   EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(scale));
   EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(shift - 1));

   EMIT(REG_DPU_EW_OP_VALUE_0, 0); EMIT(REG_DPU_EW_OP_VALUE_1, 0);
   EMIT(REG_DPU_EW_OP_VALUE_2, 0); EMIT(REG_DPU_EW_OP_VALUE_3, 0);
   EMIT(REG_DPU_EW_OP_VALUE_4, 0); EMIT(REG_DPU_EW_OP_VALUE_5, 0);
   EMIT(REG_DPU_EW_OP_VALUE_6, 0); EMIT(REG_DPU_EW_OP_VALUE_7, 0);
   EMIT(REG_DPU_SURFACE_ADD, 0);
   emit_raw(&dst, DPU | 0x1, 0x40c4, 0);
   EMIT(REG_DPU_LUT_ACCESS_CFG, 0); EMIT(REG_DPU_LUT_ACCESS_DATA, 0);
   EMIT(REG_DPU_LUT_CFG, 0); EMIT(REG_DPU_LUT_INFO, 0);
   EMIT(REG_DPU_LUT_LE_START, 0); EMIT(REG_DPU_LUT_LE_END, 0);
   EMIT(REG_DPU_LUT_LO_START, 0); EMIT(REG_DPU_LUT_LO_END, 0);
   EMIT(REG_DPU_LUT_LE_SLOPE_SCALE, 0); EMIT(REG_DPU_LUT_LE_SLOPE_SHIFT, 0);
   EMIT(REG_DPU_LUT_LO_SLOPE_SCALE, 0); EMIT(REG_DPU_LUT_LO_SLOPE_SHIFT, 0);
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(task->output_width - 1));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(task->output_height - 1));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 0);
   EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0);
   EMIT(REG_DPU_RDMA_RDMA_BRDMA_CFG, 0x1f);
   EMIT(REG_DPU_RDMA_RDMA_BS_BASE_ADDR, 0);
   EMIT(REG_DPU_RDMA_RDMA_NRDMA_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_BN_BASE_ADDR, 0);
   EMIT(REG_DPU_RDMA_RDMA_ERDMA_CFG, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(1));
   EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0);
   EMIT(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 0);
   EMIT(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG,
        DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) |
        DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(1));
   EMIT(REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH, 0);
   EMIT(REG_DPU_RDMA_RDMA_PAD_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_WEIGHT, 0);
   EMIT(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, 0);

   EMIT(REG_PC_BASE_ADDRESS, 0);
   EMIT(REG_PC_REGISTER_AMOUNTS, 0);
   *dst++ = 0x0041000000000000ULL;
   emit_raw(&dst, 0x81, REG_PC_OPERATION_ENABLE,
            PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1));

   return (unsigned)(dst - out);
}

/*
 * Hybrid regcmd: starts from standard, selectively overrides registers
 * with per-channel values based on a bitmask. Used for binary-searching
 * which register difference causes RKNPU hangs.
 *
 * Bitmask bits:
 *  0: DPU_DATA_FORMAT      → BS_MUL_SHIFT_VALUE_NEG(7) vs 0
 *  1: DPU_DATA_CUBE_CHANNEL → 0 vs ORIG|CHANNEL
 *  2: DPU_BS_CFG           → 0x13F vs ALU_ALGO|ALU_SRC|RELU_BYP|MUL_BYP
 *  3: DPU_BS_ALU_CFG       → per_channel_bias vs 0
 *  4: DPU_BS_RELUX_CMP     → 0x001F001F vs 0
 *  5: DPU_BS_OW_OP         → 0 vs 0x80-wzp
 *  6: DPU_WDMA_SIZE_0      → 0 vs CHANNEL_WDMA(oc-1)
 *  7: DPU_BN_CFG           → 0x12 vs all bypass
 *  8: DPU_EW_CFG           → 0x383 vs full bypass
 *  9: BRDMA_CFG            → 0x1F vs BRDMA_DATA_USE(1)
 * 10: BS_BASE_ADDR         → 0 vs bias_dma_addr
 * 11: DPU_SURFACE_ADD      → 0 vs surfaces_per_row
 * 12: CORE_CLIP_TRUNCATE   → 0 vs truncate_bits
 * 13: CNA_CVT_CON5         → 0 vs conditional
 * 14: CNA_PAD_CON1         → izp-0x80 vs complex logic
 * 15: RDMA_DATA_CUBE_CHANNEL → 0 vs oc-1
 * 16: CNA_CONV_CON2        → 0 vs FEATURE_GRAINS(...)
 * 17: RDMA_WEIGHT          → 0 vs weights set
 */
static unsigned fill_hybrid_regcmd(const struct rnpu_model *model,
                                    const struct rnpu_operation *op,
                                    uint64_t *out, unsigned task_num,
                                    uint32_t mask)
{
   uint64_t *dst = out;
   const struct rnpu_split_task *task = &op->tasks[task_num];
   unsigned num_tasks = op->task_count;
   unsigned ozp = task->output_zero_point;
   unsigned wzp = task->weights_zero_point;
   unsigned offset = ozp - 0x80;

   uint64_t act_base = model->activation_bo.dma_addr;
   uint64_t wt_base = model->weight_bo.dma_addr;
   uint64_t bias_base = model->bias_bo.dma_addr;

   uint32_t con0 = CNA_CBUF_CON0_WEIGHT_BANK(task->weights_banks) |
                   CNA_CBUF_CON0_DATA_BANK(task->input_banks);

   EMIT(REG_CNA_CBUF_CON0, con0);
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   EMIT(REG_CNA_DCOMP_CTRL, 0);

   uint32_t con1 = 0;
   if (task->input_channels_real == 1)
      con1 |= CNA_CONV_CON1_NONALIGN_DMA(1) | CNA_CONV_CON1_GROUP_LINE_OFF(1) |
              CNA_CONV_CON1_ARGB_IN(8);
   if (op->depthwise)
      con1 |= CNA_CONV_CON1_CONV_MODE(3);

   EMIT(REG_CNA_CONV_CON1, con1);
   EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) |
                            DPU_S_POINTER_EXECUTER_PP_EN(1) |
                            DPU_S_POINTER_POINTER_PP_EN(1));
   EMIT(REG_DPU_RDMA_RDMA_S_POINTER,
        DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) |
        DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) |
        DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1));
   EMIT(REG_CNA_CONV_CON1, con1);

   /* Bit 16: CNA_CONV_CON2 */
   if (mask & (1u << 16))
      EMIT(REG_CNA_CONV_CON2, 0);
   else
      EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_FEATURE_GRAINS(50 + task->stride_y + 1));

   EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_X_STRIDE(task->stride_x) |
                            CNA_CONV_CON3_CONV_Y_STRIDE(task->stride_y));
   EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(task->input_width) |
                             CNA_DATA_SIZE0_DATAIN_HEIGHT(task->input_height));
   EMIT(REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(task->input_channels_real - 1) |
                             CNA_DATA_SIZE1_DATAIN_CHANNEL(task->input_channels));
   EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(task->output_width));
   EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(task->atomic_count));
   EMIT(REG_CNA_WEIGHT_SIZE0, task->weights_width * task->weights_height *
                               task->input_channels * task->weights_kernels);
   EMIT(REG_CNA_WEIGHT_SIZE1, task->weights_width * task->weights_height *
                               task->input_channels);
   EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(task->weights_width) |
                               CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(task->weights_height) |
                               CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(task->weights_kernels));
   EMIT(REG_CNA_CBUF_CON0, con0);
   EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(task->input_data_entries));

   if (task->input_channels_real == 1) {
      unsigned truncate = 14, scale = 16384, cvt_off = 65408;
      if (op->addition_input || op->add_tensor != -1) { truncate = 15; scale = 32388; }
      EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_CVT_TRUNCATE_3(truncate) |
                              CNA_CVT_CON0_CVT_TRUNCATE_2(truncate) |
                              CNA_CVT_CON0_CVT_TRUNCATE_1(truncate) |
                              CNA_CVT_CON0_CVT_TRUNCATE_0(truncate));
      EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(scale) | CNA_CVT_CON1_CVT_OFFSET0(cvt_off));
      EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(scale) | CNA_CVT_CON2_CVT_OFFSET1(cvt_off));
      EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(scale) | CNA_CVT_CON3_CVT_OFFSET2(cvt_off));
      EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(scale) | CNA_CVT_CON4_CVT_OFFSET3(cvt_off));
   } else {
      EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_DATA_SIGN(1) | CNA_CVT_CON0_CVT_TYPE(1) |
                              CNA_CVT_CON0_CVT_BYPASS(1));
      EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
      EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
      EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
      EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
   }

   EMIT(REG_CNA_FC_CON0, 0);
   EMIT(REG_CNA_FC_CON1, 0);
   EMIT(REG_CNA_PAD_CON0, CNA_PAD_CON0_PAD_LEFT(task->pad_left) |
                           CNA_PAD_CON0_PAD_TOP(task->pad_top));

   uint32_t in_addr = (uint32_t)(act_base + model->tensors[op->input_tensor].offset +
                                 task->input_offset);
   EMIT(REG_CNA_FEATURE_DATA_ADDR, in_addr);
   EMIT(REG_CNA_FC_CON2, 0);
   EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) |
                           CNA_DMA_CON0_DATA_BURST_LEN(15));
   EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(task->input_line_stride));
   EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(task->input_surface_stride));
   EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(op->input_width) |
                                CNA_FC_DATA_SIZE0_DMA_HEIGHT(task->input_height));
   EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(task->input_channels));
   EMIT(REG_CNA_DCOMP_CTRL, 0);
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   EMIT(REG_CNA_DCOMP_ADDR0, (uint32_t)(wt_base + op->weight_offset));
   EMIT(REG_CNA_DCOMP_AMOUNT0, 0); EMIT(REG_CNA_DCOMP_AMOUNT1, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT2, 0); EMIT(REG_CNA_DCOMP_AMOUNT3, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT4, 0); EMIT(REG_CNA_DCOMP_AMOUNT5, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT6, 0); EMIT(REG_CNA_DCOMP_AMOUNT7, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT8, 0); EMIT(REG_CNA_DCOMP_AMOUNT9, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT10, 0); EMIT(REG_CNA_DCOMP_AMOUNT11, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT12, 0); EMIT(REG_CNA_DCOMP_AMOUNT13, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT14, 0); EMIT(REG_CNA_DCOMP_AMOUNT15, 0);

   /* Bit 13: CNA_CVT_CON5 */
   if (mask & (1u << 13))
      EMIT(REG_CNA_CVT_CON5, 0);
   else
      EMIT(REG_CNA_CVT_CON5, task->input_channels_real == 1 ? 65535 : 0);

   /* Bit 14: CNA_PAD_CON1 */
   if (mask & (1u << 14)) {
      EMIT(REG_CNA_PAD_CON1, task->input_zero_point - 0x80);
   } else {
      int32_t pad_con1;
      if (task->weights_width >= 3 && task->input_zero_point == 0x0)
         pad_con1 = 0xffff8080;
      else
         pad_con1 = task->input_zero_point - 0x80;
      if (op->addition_input || op->add_tensor != -1)
         pad_con1 = 0xffffff80;
      if (op->depthwise && task->input_zero_point == 0x8b)
         pad_con1 = 0x0b0b;
      EMIT(REG_CNA_PAD_CON1, pad_con1);
   }

   uint32_t misc_cfg = CORE_MISC_CFG_QD_EN(1);
   if (op->depthwise) misc_cfg |= CORE_MISC_CFG_DW_EN(1);
   EMIT(REG_CORE_MISC_CFG, misc_cfg);
   EMIT(REG_CORE_DATAOUT_SIZE_0,
        CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(task->output_height - 1) |
        CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(task->output_width - 1));
   EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(task->output_channels - 1));

   /* Bit 12: CORE_CLIP_TRUNCATE */
   if (mask & (1u << 12))
      EMIT(REG_CORE_CLIP_TRUNCATE, 0);
   else
      EMIT(REG_CORE_CLIP_TRUNCATE, CORE_CLIP_TRUNCATE_CLIP_TRUNCATE(op->truncate_bits));

   emit_raw(&dst, CORE | 0x1, 0x3030, 0);

   uint32_t fmc = DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2);
   if (op->depthwise) fmc |= DPU_FEATURE_MODE_CFG_CONV_MODE(3);
   EMIT(REG_DPU_FEATURE_MODE_CFG, fmc);

   /* Bit 0: DPU_DATA_FORMAT */
   if (mask & (1u << 0))
      EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_BS_MUL_SHIFT_VALUE_NEG(7));
   else
      EMIT(REG_DPU_DATA_FORMAT, 0);

   EMIT(REG_DPU_OFFSET_PEND, 0);

   uint32_t out_addr = (uint32_t)(act_base + model->tensors[op->output_tensor].offset +
                                  task->output_offset + op->per_channel_group_offset);
   EMIT(REG_DPU_DST_BASE_ADDR, out_addr);
   EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(task->output_surface_stride));
   EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(task->output_width - 1));
   EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(task->output_height - 1));
   EMIT(REG_DPU_DATA_CUBE_NOTCH_ADDR, 0);

   /* Bit 1: DPU_DATA_CUBE_CHANNEL */
   if (mask & (1u << 1))
      EMIT(REG_DPU_DATA_CUBE_CHANNEL, 0);
   else
      EMIT(REG_DPU_DATA_CUBE_CHANNEL,
           DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(task->output_channels_real - 1) |
           DPU_DATA_CUBE_CHANNEL_CHANNEL(task->output_channels - 1));

   /* Bit 2: DPU_BS_CFG
    * Per-channel mode: 0x13F matches RKNN vendor (proven to work on 2conv).
    * BS_BYPASS=1 but hardware still uses scalar BS_ALU_CFG via undocumented
    * bits 2-3 — this is the mechanism RKNN uses for per-channel bias. */
   if (mask & (1u << 2))
      EMIT(REG_DPU_BS_CFG, 0x13f);
   else
      EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_ALU_ALGO(2) | DPU_BS_CFG_BS_ALU_SRC(1) |
                            DPU_BS_CFG_BS_RELU_BYPASS(1) | DPU_BS_CFG_BS_MUL_BYPASS(1));

   /* Bit 3: DPU_BS_ALU_CFG */
   if (mask & (1u << 3))
      EMIT(REG_DPU_BS_ALU_CFG, (uint32_t)op->per_channel_bias);
   else
      EMIT(REG_DPU_BS_ALU_CFG, 0);

   EMIT(REG_DPU_BS_MUL_CFG, 0);

   /* Bit 4: DPU_BS_RELUX_CMP_VALUE */
   if (mask & (1u << 4))
      EMIT(REG_DPU_BS_RELUX_CMP_VALUE, 0x001f001f);
   else
      EMIT(REG_DPU_BS_RELUX_CMP_VALUE, 0);

   if (op->depthwise) {
      EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(3) |
                               DPU_BS_OW_CFG_SIZE_E_1(3) |
                               DPU_BS_OW_CFG_SIZE_E_0(3));
   } else {
      EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(1) |
                               DPU_BS_OW_CFG_SIZE_E_1(1) |
                               DPU_BS_OW_CFG_SIZE_E_0(1));
   }

   /* Bit 5: DPU_BS_OW_OP */
   if (mask & (1u << 5))
      EMIT(REG_DPU_BS_OW_OP, 0);
   else
      EMIT(REG_DPU_BS_OW_OP, 0);

   /* Bit 6: DPU_WDMA_SIZE_0 */
   if (mask & (1u << 6))
      EMIT(REG_DPU_WDMA_SIZE_0, 0);
   else
      EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(task->output_channels - 1));

   EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(task->output_height - 1) |
                              DPU_WDMA_SIZE_1_WIDTH_WDMA(task->output_width - 1));

   /* Bit 7: DPU_BN_CFG */
   if (mask & (1u << 7))
      EMIT(REG_DPU_BN_CFG, 0x12);
   else
      EMIT(REG_DPU_BN_CFG, DPU_BN_CFG_BN_RELU_BYPASS(1) | DPU_BN_CFG_BN_MUL_BYPASS(1) |
                            DPU_BN_CFG_BN_ALU_BYPASS(1) | DPU_BN_CFG_BN_BYPASS(1));

   EMIT(REG_DPU_BN_ALU_CFG, 0);
   EMIT(REG_DPU_BN_MUL_CFG, 0);
   EMIT(REG_DPU_BN_RELUX_CMP_VALUE, 0);

   /* Bit 8: DPU_EW_CFG */
   if (mask & (1u << 8))
      EMIT(REG_DPU_EW_CFG, 0x383);
   else
      EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) |
                            DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) |
                            DPU_EW_CFG_EW_BYPASS(1));

   EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, 0);
   EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
   EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0);

   float conv_scale = (task->input_scale * task->weights_scale) / task->output_scale;
   uint32_t scale_bits = fui(conv_scale);
   unsigned shift = 127 + 31 - 32 - (scale_bits >> 23) + 16;
   if (!(mask & (1u << 12)) && op->truncate_bits > 0) shift--;
   unsigned scale = ((scale_bits >> 9) & 0x7fff) + 1;
   if (scale < 1 << 14) scale |= 1 << 14;

   /* When BS is bypassed (bit 2 set), fold bias into OUT_CVT_OFFSET.
    * The NPU ignores BS_ALU_CFG when BS_BYPASS=1, so we must account
    * for bias in the output conversion offset instead.
    * Use hardware scale/shift for exact integer computation:
    *   bias_out = (bias * scale) >> (shift - 1)
    * This matches the hardware OUT_CVT computation. */
   int32_t final_offset = (int32_t)offset;
   if ((mask & (1u << 2)) && op->per_channel_bias != 0) {
      int64_t bias_scaled = (int64_t)op->per_channel_bias * (int64_t)scale;
      int32_t bias_contribution = (int32_t)nvdla_shift_right_round64(bias_scaled, shift - 1);
      final_offset += bias_contribution;
   }
   EMIT(REG_DPU_OUT_CVT_OFFSET, (uint32_t)final_offset);

   EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(scale));
   EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(shift - 1));

   EMIT(REG_DPU_EW_OP_VALUE_0, 0); EMIT(REG_DPU_EW_OP_VALUE_1, 0);
   EMIT(REG_DPU_EW_OP_VALUE_2, 0); EMIT(REG_DPU_EW_OP_VALUE_3, 0);
   EMIT(REG_DPU_EW_OP_VALUE_4, 0); EMIT(REG_DPU_EW_OP_VALUE_5, 0);
   EMIT(REG_DPU_EW_OP_VALUE_6, 0); EMIT(REG_DPU_EW_OP_VALUE_7, 0);

   /* Bit 11: DPU_SURFACE_ADD */
   if (mask & (1u << 11))
      EMIT(REG_DPU_SURFACE_ADD, 0);
   else
      EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(task->surfaces_per_row));

   emit_raw(&dst, DPU | 0x1, 0x40c4, 0);
   EMIT(REG_DPU_LUT_ACCESS_CFG, 0); EMIT(REG_DPU_LUT_ACCESS_DATA, 0);
   EMIT(REG_DPU_LUT_CFG, 0); EMIT(REG_DPU_LUT_INFO, 0);
   EMIT(REG_DPU_LUT_LE_START, 0); EMIT(REG_DPU_LUT_LE_END, 0);
   EMIT(REG_DPU_LUT_LO_START, 0); EMIT(REG_DPU_LUT_LO_END, 0);
   EMIT(REG_DPU_LUT_LE_SLOPE_SCALE, 0); EMIT(REG_DPU_LUT_LE_SLOPE_SHIFT, 0);
   EMIT(REG_DPU_LUT_LO_SLOPE_SCALE, 0); EMIT(REG_DPU_LUT_LO_SLOPE_SHIFT, 0);

   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(task->output_width - 1));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(task->output_height - 1));

   /* Bit 15: RDMA_DATA_CUBE_CHANNEL */
   if (mask & (1u << 15))
      EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, 0);
   else
      EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(task->output_channels - 1));

   EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0);

   /* Bit 9: BRDMA_CFG */
   if (mask & (1u << 9))
      EMIT(REG_DPU_RDMA_RDMA_BRDMA_CFG, 0x1f);
   else
      EMIT(REG_DPU_RDMA_RDMA_BRDMA_CFG, DPU_RDMA_RDMA_BRDMA_CFG_BRDMA_DATA_USE(1));

   /* Bit 10: BS_BASE_ADDR */
   if (mask & (1u << 10))
      EMIT(REG_DPU_RDMA_RDMA_BS_BASE_ADDR, 0);
   else
      EMIT(REG_DPU_RDMA_RDMA_BS_BASE_ADDR, (uint32_t)(bias_base + op->bias_offset));

   EMIT(REG_DPU_RDMA_RDMA_NRDMA_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_BN_BASE_ADDR, 0);
   EMIT(REG_DPU_RDMA_RDMA_ERDMA_CFG, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(1));
   EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0);
   EMIT(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 0);

   uint32_t rfmc = DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15) |
                   DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(1);
   if (op->depthwise)
      rfmc |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE(3);
   EMIT(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rfmc);
   EMIT(REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH, 0);
   EMIT(REG_DPU_RDMA_RDMA_PAD_CFG, 0);

   /* Bit 17: RDMA_WEIGHT */
   if (mask & (1u << 17))
      EMIT(REG_DPU_RDMA_RDMA_WEIGHT, 0);
   else
      EMIT(REG_DPU_RDMA_RDMA_WEIGHT,
           DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(1) |
           DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(1));

   EMIT(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, 0);

   EMIT(REG_PC_BASE_ADDRESS, 0);
   EMIT(REG_PC_REGISTER_AMOUNTS, 0);
   *dst++ = 0x0041000000000000ULL;
   emit_raw(&dst, 0x81, REG_PC_OPERATION_ENABLE,
            PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1));

   return (unsigned)(dst - out);
}

/*
 * BRDMA per-channel regcmd: full conv (all OC), per-channel requantization
 * via BS MUL with DMA source. Matches RKNN register configuration.
 *
 * Key differences from standard regcmd:
 *   BS_CFG     = 0x20140 (MUL active, ALU active, ALU_SRC=DMA)
 *   BS_MUL_CFG = 0xe01   (MUL_SRC=DMA, shift=14)
 *   BRDMA_CFG  = 0xe     (DATA_USE=7: bias+mul from DMA)
 *   DATA_FORMAT = 0xe0   (BS_MUL_SHIFT_VALUE_NEG=14)
 *   BN_CFG     = 0x12    (ReLU only, no MUL/ALU)
 *   BS_BASE_ADDR → brdma_bo (combined bias+mul data)
 *   OUT_CVT_SCALE/SHIFT = uniform (from max_conv_scale)
 */
static unsigned fill_brdma_per_channel_regcmd(const struct rnpu_model *model,
                                               const struct rnpu_operation *op,
                                               uint64_t *out, unsigned task_num)
{
   uint64_t *dst = out;
   const struct rnpu_split_task *task = &op->tasks[task_num];
   unsigned ozp = task->output_zero_point;
   unsigned wzp = task->weights_zero_point;

   uint64_t act_base = model->activation_bo.dma_addr;
   uint64_t wt_base = model->weight_bo.dma_addr;
   uint64_t brdma_base = model->brdma_bo.dma_addr;

   uint32_t con0 = CNA_CBUF_CON0_WEIGHT_BANK(task->weights_banks) |
                   CNA_CBUF_CON0_DATA_BANK(task->input_banks);
   if (task_num > 0 && op->reuse_weights_cbuf)
      con0 |= CNA_CBUF_CON0_WEIGHT_REUSE(1);

   EMIT(REG_CNA_CBUF_CON0, con0);
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   EMIT(REG_CNA_DCOMP_CTRL, 0);

   uint32_t con1 = 0;
   if (task->input_channels_real == 1)
      con1 |= CNA_CONV_CON1_NONALIGN_DMA(1) | CNA_CONV_CON1_GROUP_LINE_OFF(1) |
              CNA_CONV_CON1_ARGB_IN(8);
   if (op->depthwise)
      con1 |= CNA_CONV_CON1_CONV_MODE(3);

   EMIT(REG_CNA_CONV_CON1, con1);
   EMIT(REG_DPU_S_POINTER, DPU_S_POINTER_POINTER_PP_MODE(1) |
                            DPU_S_POINTER_EXECUTER_PP_EN(1) |
                            DPU_S_POINTER_POINTER_PP_EN(1));
   EMIT(REG_DPU_RDMA_RDMA_S_POINTER,
        DPU_RDMA_RDMA_S_POINTER_POINTER_PP_MODE(1) |
        DPU_RDMA_RDMA_S_POINTER_EXECUTER_PP_EN(1) |
        DPU_RDMA_RDMA_S_POINTER_POINTER_PP_EN(1));
   EMIT(REG_CNA_CONV_CON1, con1);
   EMIT(REG_CNA_CONV_CON2, CNA_CONV_CON2_FEATURE_GRAINS(50 + task->stride_y + 1));
   EMIT(REG_CNA_CONV_CON3, CNA_CONV_CON3_CONV_X_STRIDE(task->stride_x) |
                            CNA_CONV_CON3_CONV_Y_STRIDE(task->stride_y));
   EMIT(REG_CNA_DATA_SIZE0, CNA_DATA_SIZE0_DATAIN_WIDTH(task->input_width) |
                             CNA_DATA_SIZE0_DATAIN_HEIGHT(task->input_height));
   EMIT(REG_CNA_DATA_SIZE1, CNA_DATA_SIZE1_DATAIN_CHANNEL_REAL(task->input_channels_real - 1) |
                             CNA_DATA_SIZE1_DATAIN_CHANNEL(task->input_channels));
   EMIT(REG_CNA_DATA_SIZE2, CNA_DATA_SIZE2_DATAOUT_WIDTH(task->output_width));
   EMIT(REG_CNA_DATA_SIZE3, CNA_DATA_SIZE3_DATAOUT_ATOMICS(task->atomic_count));
   EMIT(REG_CNA_WEIGHT_SIZE0, task->weights_width * task->weights_height *
                               task->input_channels * task->weights_kernels);
   EMIT(REG_CNA_WEIGHT_SIZE1, task->weights_width * task->weights_height *
                               task->input_channels);
   EMIT(REG_CNA_WEIGHT_SIZE2, CNA_WEIGHT_SIZE2_WEIGHT_WIDTH(task->weights_width) |
                               CNA_WEIGHT_SIZE2_WEIGHT_HEIGHT(task->weights_height) |
                               CNA_WEIGHT_SIZE2_WEIGHT_KERNELS(task->weights_kernels));
   EMIT(REG_CNA_CBUF_CON0, con0);
   EMIT(REG_CNA_CBUF_CON1, CNA_CBUF_CON1_DATA_ENTRIES(task->input_data_entries));

   if (task->input_channels_real == 1) {
      unsigned truncate = 14, scale = 16384, cvt_off = 65408;
      EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_CVT_TRUNCATE_3(truncate) |
                              CNA_CVT_CON0_CVT_TRUNCATE_2(truncate) |
                              CNA_CVT_CON0_CVT_TRUNCATE_1(truncate) |
                              CNA_CVT_CON0_CVT_TRUNCATE_0(truncate));
      EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(scale) | CNA_CVT_CON1_CVT_OFFSET0(cvt_off));
      EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(scale) | CNA_CVT_CON2_CVT_OFFSET1(cvt_off));
      EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(scale) | CNA_CVT_CON3_CVT_OFFSET2(cvt_off));
      EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(scale) | CNA_CVT_CON4_CVT_OFFSET3(cvt_off));
   } else {
      EMIT(REG_CNA_CVT_CON0, CNA_CVT_CON0_DATA_SIGN(1) | CNA_CVT_CON0_CVT_TYPE(1) |
                              CNA_CVT_CON0_CVT_BYPASS(1));
      EMIT(REG_CNA_CVT_CON1, CNA_CVT_CON1_CVT_SCALE0(1));
      EMIT(REG_CNA_CVT_CON2, CNA_CVT_CON2_CVT_SCALE1(1));
      EMIT(REG_CNA_CVT_CON3, CNA_CVT_CON3_CVT_SCALE2(1));
      EMIT(REG_CNA_CVT_CON4, CNA_CVT_CON4_CVT_SCALE3(1));
   }

   EMIT(REG_CNA_FC_CON0, 0);
   EMIT(REG_CNA_FC_CON1, 0);
   EMIT(REG_CNA_PAD_CON0, CNA_PAD_CON0_PAD_LEFT(task->pad_left) |
                           CNA_PAD_CON0_PAD_TOP(task->pad_top));

   uint32_t in_addr = (uint32_t)(act_base + model->tensors[op->input_tensor].offset +
                                 task->input_offset);
   EMIT(REG_CNA_FEATURE_DATA_ADDR, in_addr);
   EMIT(REG_CNA_FC_CON2, 0);
   EMIT(REG_CNA_DMA_CON0, CNA_DMA_CON0_WEIGHT_BURST_LEN(15) |
                           CNA_DMA_CON0_DATA_BURST_LEN(15));
   EMIT(REG_CNA_DMA_CON1, CNA_DMA_CON1_LINE_STRIDE(task->input_line_stride));
   EMIT(REG_CNA_DMA_CON2, CNA_DMA_CON2_SURF_STRIDE(task->input_surface_stride));
   EMIT(REG_CNA_FC_DATA_SIZE0, CNA_FC_DATA_SIZE0_DMA_WIDTH(op->input_width) |
                                CNA_FC_DATA_SIZE0_DMA_HEIGHT(task->input_height));
   EMIT(REG_CNA_FC_DATA_SIZE1, CNA_FC_DATA_SIZE1_DMA_CHANNEL(task->input_channels));
   EMIT(REG_CNA_DCOMP_CTRL, 0);
   EMIT(REG_CNA_DCOMP_REGNUM, 0);
   EMIT(REG_CNA_DCOMP_ADDR0, (uint32_t)(wt_base + op->weight_offset));
   EMIT(REG_CNA_DCOMP_AMOUNT0, 0); EMIT(REG_CNA_DCOMP_AMOUNT1, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT2, 0); EMIT(REG_CNA_DCOMP_AMOUNT3, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT4, 0); EMIT(REG_CNA_DCOMP_AMOUNT5, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT6, 0); EMIT(REG_CNA_DCOMP_AMOUNT7, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT8, 0); EMIT(REG_CNA_DCOMP_AMOUNT9, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT10, 0); EMIT(REG_CNA_DCOMP_AMOUNT11, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT12, 0); EMIT(REG_CNA_DCOMP_AMOUNT13, 0);
   EMIT(REG_CNA_DCOMP_AMOUNT14, 0); EMIT(REG_CNA_DCOMP_AMOUNT15, 0);

   EMIT(REG_CNA_CVT_CON5, task->input_channels_real == 1 ? 65535 : 0);

   int32_t pad_con1;
   if (task->weights_width >= 3 && task->input_zero_point == 0x0)
      pad_con1 = 0xffff8080;
   else
      pad_con1 = task->input_zero_point - 0x80;
   EMIT(REG_CNA_PAD_CON1, pad_con1);

   uint32_t misc_cfg = CORE_MISC_CFG_QD_EN(1);
   if (op->depthwise) misc_cfg |= CORE_MISC_CFG_DW_EN(1);
   EMIT(REG_CORE_MISC_CFG, misc_cfg);
   EMIT(REG_CORE_DATAOUT_SIZE_0,
        CORE_DATAOUT_SIZE_0_DATAOUT_HEIGHT(task->output_height - 1) |
        CORE_DATAOUT_SIZE_0_DATAOUT_WIDTH(task->output_width - 1));
   EMIT(REG_CORE_DATAOUT_SIZE_1, CORE_DATAOUT_SIZE_1_DATAOUT_CHANNEL(task->output_channels - 1));
   EMIT(REG_CORE_CLIP_TRUNCATE, CORE_CLIP_TRUNCATE_CLIP_TRUNCATE(op->truncate_bits));
   emit_raw(&dst, CORE | 0x1, 0x3030, 0);

   uint32_t fmc = DPU_FEATURE_MODE_CFG_BURST_LEN(15) | DPU_FEATURE_MODE_CFG_OUTPUT_MODE(2);
   if (op->depthwise) fmc |= DPU_FEATURE_MODE_CFG_CONV_MODE(3);
   EMIT(REG_DPU_FEATURE_MODE_CFG, fmc);

   /* DATA_FORMAT: BS_MUL_SHIFT_VALUE_NEG from dynamic mul_shift */
   EMIT(REG_DPU_DATA_FORMAT, DPU_DATA_FORMAT_BS_MUL_SHIFT_VALUE_NEG(op->brdma_mul_shift));

   EMIT(REG_DPU_OFFSET_PEND, 0);

   uint32_t out_addr = (uint32_t)(act_base + model->tensors[op->output_tensor].offset +
                                  task->output_offset);
   EMIT(REG_DPU_DST_BASE_ADDR, out_addr);
   EMIT(REG_DPU_DST_SURF_STRIDE, DPU_DST_SURF_STRIDE_DST_SURF_STRIDE(task->output_surface_stride));
   EMIT(REG_DPU_DATA_CUBE_WIDTH, DPU_DATA_CUBE_WIDTH_WIDTH(task->output_width - 1));
   EMIT(REG_DPU_DATA_CUBE_HEIGHT, DPU_DATA_CUBE_HEIGHT_HEIGHT(task->output_height - 1));
   EMIT(REG_DPU_DATA_CUBE_NOTCH_ADDR, 0);
   EMIT(REG_DPU_DATA_CUBE_CHANNEL,
        DPU_DATA_CUBE_CHANNEL_ORIG_CHANNEL(task->output_channels_real - 1) |
        DPU_DATA_CUBE_CHANNEL_CHANNEL(task->output_channels - 1));

   /* BS_CFG: ALU=ADD, ALU_SRC=DMA, MUL active, ReLU bypassed (BN handles ReLU)
    * 0x20140 = BS_ALU_ALGO(2) | BS_ALU_SRC(1) | BS_RELU_BYPASS(1) */
   EMIT(REG_DPU_BS_CFG, DPU_BS_CFG_BS_ALU_ALGO(2) | DPU_BS_CFG_BS_ALU_SRC(1) |
                         DPU_BS_CFG_BS_RELU_BYPASS(1));
   EMIT(REG_DPU_BS_ALU_CFG, 0);

   /* BS_MUL_CFG: MUL_SRC=DMA (bit 0), shift from brdma_mul_shift */
   EMIT(REG_DPU_BS_MUL_CFG, (op->brdma_mul_shift << 8) | 1);
   EMIT(REG_DPU_BS_RELUX_CMP_VALUE, 0);

   if (op->depthwise) {
      EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(3) |
                               DPU_BS_OW_CFG_SIZE_E_1(3) |
                               DPU_BS_OW_CFG_SIZE_E_0(3));
   } else {
      EMIT(REG_DPU_BS_OW_CFG, DPU_BS_OW_CFG_SIZE_E_2(1) |
                               DPU_BS_OW_CFG_SIZE_E_1(1) |
                               DPU_BS_OW_CFG_SIZE_E_0(1));
   }
   EMIT(REG_DPU_BS_OW_OP, 0);
   EMIT(REG_DPU_WDMA_SIZE_0, DPU_WDMA_SIZE_0_CHANNEL_WDMA(task->output_channels - 1));
   EMIT(REG_DPU_WDMA_SIZE_1, DPU_WDMA_SIZE_1_HEIGHT_WDMA(task->output_height - 1) |
                              DPU_WDMA_SIZE_1_WIDTH_WDMA(task->output_width - 1));

   /* BN_CFG: ReLU only (matches RKNN: 0x12 = alu_bypass + mul_bypass + relu active) */
   EMIT(REG_DPU_BN_CFG, 0x12);
   EMIT(REG_DPU_BN_ALU_CFG, 0);
   EMIT(REG_DPU_BN_MUL_CFG, 0);
   EMIT(REG_DPU_BN_RELUX_CMP_VALUE, 0);

   /* EW (element-wise addition) — mirrors standard regcmd's ADD handling */
   if (op->add_tensor != -1) {
      EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_CVT_TYPE(1) | DPU_EW_CFG_EW_DATA_MODE(1) |
                            DPU_EW_CFG_EDATA_SIZE(1) | DPU_EW_CFG_EW_ALU_ALGO(2) |
                            DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_LUT_BYPASS(1) |
                            DPU_EW_CFG_EW_OP_SRC(1));
      EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, op->addition_offset);
      EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
      EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0);
   } else {
      EMIT(REG_DPU_EW_CFG, DPU_EW_CFG_EW_RELU_BYPASS(1) | DPU_EW_CFG_EW_OP_CVT_BYPASS(1) |
                            DPU_EW_CFG_EW_LUT_BYPASS(1) | DPU_EW_CFG_EW_OP_BYPASS(1) |
                            DPU_EW_CFG_EW_BYPASS(1));
      EMIT(REG_DPU_EW_CVT_OFFSET_VALUE, 0);
      EMIT(REG_DPU_EW_CVT_SCALE_VALUE, DPU_EW_CVT_SCALE_VALUE_EW_OP_CVT_SCALE(1));
      EMIT(REG_DPU_EW_RELUX_CMP_VALUE, 0);
   }

   /* OUT_CVT: uniform scale from max_conv_scale */
   int offset = (int)ozp - 0x80;
   float conv_scale = (task->input_scale * task->weights_scale) / task->output_scale;
   uint32_t scale_bits = fui(conv_scale);
   unsigned shift = 127 + 31 - 32 - (scale_bits >> 23) + 16;
   if (op->truncate_bits > 0) shift--;
   unsigned scale = ((scale_bits >> 9) & 0x7fff) + 1;
   if (scale < 1 << 14) scale |= 1 << 14;

   EMIT(REG_DPU_OUT_CVT_OFFSET, (uint32_t)offset);
   EMIT(REG_DPU_OUT_CVT_SCALE, DPU_OUT_CVT_SCALE_OUT_CVT_SCALE(scale));
   EMIT(REG_DPU_OUT_CVT_SHIFT, DPU_OUT_CVT_SHIFT_OUT_CVT_SHIFT(shift - 1));

   EMIT(REG_DPU_EW_OP_VALUE_0, 0); EMIT(REG_DPU_EW_OP_VALUE_1, 0);
   EMIT(REG_DPU_EW_OP_VALUE_2, 0); EMIT(REG_DPU_EW_OP_VALUE_3, 0);
   EMIT(REG_DPU_EW_OP_VALUE_4, 0); EMIT(REG_DPU_EW_OP_VALUE_5, 0);
   EMIT(REG_DPU_EW_OP_VALUE_6, 0); EMIT(REG_DPU_EW_OP_VALUE_7, 0);
   EMIT(REG_DPU_SURFACE_ADD, DPU_SURFACE_ADD_SURF_ADD(task->surfaces_per_row));
   emit_raw(&dst, DPU | 0x1, 0x40c4, 0);
   EMIT(REG_DPU_LUT_ACCESS_CFG, 0); EMIT(REG_DPU_LUT_ACCESS_DATA, 0);
   EMIT(REG_DPU_LUT_CFG, 0); EMIT(REG_DPU_LUT_INFO, 0);
   EMIT(REG_DPU_LUT_LE_START, 0); EMIT(REG_DPU_LUT_LE_END, 0);
   EMIT(REG_DPU_LUT_LO_START, 0); EMIT(REG_DPU_LUT_LO_END, 0);
   EMIT(REG_DPU_LUT_LE_SLOPE_SCALE, 0); EMIT(REG_DPU_LUT_LE_SLOPE_SHIFT, 0);
   EMIT(REG_DPU_LUT_LO_SLOPE_SCALE, 0); EMIT(REG_DPU_LUT_LO_SLOPE_SHIFT, 0);

   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_WIDTH, DPU_RDMA_RDMA_DATA_CUBE_WIDTH_WIDTH(task->output_width - 1));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_HEIGHT, DPU_RDMA_RDMA_DATA_CUBE_HEIGHT_HEIGHT(task->output_height - 1));
   EMIT(REG_DPU_RDMA_RDMA_DATA_CUBE_CHANNEL, DPU_RDMA_RDMA_DATA_CUBE_CHANNEL_CHANNEL(task->output_channels - 1));

   if (op->add_tensor != -1) {
      uint32_t add_addr = (uint32_t)(act_base + model->tensors[op->add_tensor].offset +
                                     task->output_offset);
      EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, add_addr);
   } else {
      EMIT(REG_DPU_RDMA_RDMA_SRC_BASE_ADDR, 0);
   }

   /* BRDMA_CFG: DATA_USE=7 → loads bias + MUL data (0x0e = DATA_USE(7) << 1) */
   EMIT(REG_DPU_RDMA_RDMA_BRDMA_CFG, 0x0e);

   /* BS_BASE_ADDR: points to BRDMA data. For requant groups, each task points
    * to its group's BRDMA buffer via brdma_group_offset. */
   if (task->brdma_group_offset)
      EMIT(REG_DPU_RDMA_RDMA_BS_BASE_ADDR, (uint32_t)(brdma_base + task->brdma_group_offset));
   else
      EMIT(REG_DPU_RDMA_RDMA_BS_BASE_ADDR, (uint32_t)(brdma_base + op->brdma_offset));

   EMIT(REG_DPU_RDMA_RDMA_NRDMA_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_BN_BASE_ADDR, 0);

   unsigned ew_stride = MAX2(op->output_width * op->output_height, 12);

   if (op->add_tensor != -1) {
      EMIT(REG_DPU_RDMA_RDMA_ERDMA_CFG,
           DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_MODE(1) |
           DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DATA_SIZE(1));
      uint32_t ew_base = op->output_width * op->output_height * ATOMIC_K_SIZE;
      uint32_t ew_addr = (uint32_t)(act_base + model->tensors[op->add_tensor].offset +
                                    task->output_offset + ew_base);
      EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, ew_addr);
      EMIT(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE,
           DPU_RDMA_RDMA_EW_SURF_STRIDE_EW_SURF_STRIDE(ew_stride));
   } else {
      EMIT(REG_DPU_RDMA_RDMA_ERDMA_CFG, DPU_RDMA_RDMA_ERDMA_CFG_ERDMA_DISABLE(1));
      EMIT(REG_DPU_RDMA_RDMA_EW_BASE_ADDR, 0);
      EMIT(REG_DPU_RDMA_RDMA_EW_SURF_STRIDE, 0);
   }

   uint32_t rfmc = DPU_RDMA_RDMA_FEATURE_MODE_CFG_BURST_LEN(15);
   if (op->add_tensor != -1)
      rfmc |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_COMB_USE(5);
   else
      rfmc |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_MRDMA_DISABLE(1);
   if (op->depthwise)
      rfmc |= DPU_RDMA_RDMA_FEATURE_MODE_CFG_CONV_MODE(3);
   EMIT(REG_DPU_RDMA_RDMA_FEATURE_MODE_CFG, rfmc);
   EMIT(REG_DPU_RDMA_RDMA_SRC_DMA_CFG, 0);

   unsigned sn = ew_stride + task->output_width * (op->output_height - task->output_height);
   if (op->input_width == 3) sn = 15;
   if (op->add_tensor != -1) {
      EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH, DPU_RDMA_RDMA_SURF_NOTCH_SURF_NOTCH_ADDR(sn));
   } else {
      EMIT(REG_DPU_RDMA_RDMA_SURF_NOTCH, 0);
   }

   EMIT(REG_DPU_RDMA_RDMA_PAD_CFG, 0);
   EMIT(REG_DPU_RDMA_RDMA_WEIGHT,
        DPU_RDMA_RDMA_WEIGHT_E_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_N_WEIGHT(1) |
        DPU_RDMA_RDMA_WEIGHT_B_WEIGHT(1) | DPU_RDMA_RDMA_WEIGHT_M_WEIGHT(1));

   if (op->add_tensor != -1) {
      EMIT(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, DPU_RDMA_RDMA_EW_SURF_NOTCH_EW_SURF_NOTCH(sn));
   } else {
      EMIT(REG_DPU_RDMA_RDMA_EW_SURF_NOTCH, 0);
   }

   EMIT(REG_PC_BASE_ADDRESS, 0);
   EMIT(REG_PC_REGISTER_AMOUNTS, 0);
   *dst++ = 0x0041000000000000ULL;
   emit_raw(&dst, 0x81, REG_PC_OPERATION_ENABLE,
            PC_OPERATION_ENABLE_RESERVED_0(14) | PC_OPERATION_ENABLE_OP_EN(1));

   return (unsigned)(dst - out);
}

/* Global hybrid mask — set via RNPU_HYBRID_MASK env var or programmatically.
 * 0 = standard path, 0x3FFFF = all per-channel, -1 = use standard function. */
uint32_t rnpu_hybrid_mask = UINT32_MAX;

unsigned rnpu_fill_regcmd(const struct rnpu_model *model,
                          const struct rnpu_operation *op,
                          uint64_t *dst, unsigned max_regs,
                          unsigned task_num)
{
   /* RKNPU full-conv per-channel: uses BRDMA with bias+MUL DMA data */
   if (rnpu_active_driver == RNPU_DRIVER_RKNPU && op->use_brdma_per_channel)
      return fill_brdma_per_channel_regcmd(model, op, dst, task_num);

   /* For RKNPU per-channel ops (GS=1), use per-channel regcmd with
    * BN-stage bias addition. Standard regcmd doesn't work for GS=1 on RKNPU. */
   if (rnpu_active_driver == RNPU_DRIVER_RKNPU &&
       op->output_channels == 1 && op->output_tensor_channels > 0)
      return fill_per_channel_regcmd(model, op, dst, task_num);

   if (rnpu_hybrid_mask == UINT32_MAX)
      return fill_standard_regcmd(model, op, dst, task_num);

   /* Only apply per-channel mask to single-channel ops (GS=1 decomposed).
    * Per-tensor ops need standard bias DMA, so always use standard for them. */
   if (op->output_channels > 1 || op->output_tensor_channels == 0)
      return fill_standard_regcmd(model, op, dst, task_num);

   return fill_hybrid_regcmd(model, op, dst, task_num, rnpu_hybrid_mask);
}
