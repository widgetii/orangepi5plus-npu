/*
 * openrknn — Minimal FlatBuffer reader (from scratch)
 *
 * SPDX-License-Identifier: MIT
 */
#include "openrknn.h"
#include <string.h>

/* ======================================================================
 * Primitive readers — little-endian
 * ====================================================================== */

uint16_t orknn_fb_u16(const uint8_t *b, uint32_t p)
{
    return (uint16_t)b[p] | ((uint16_t)b[p + 1] << 8);
}

uint32_t orknn_fb_u32(const uint8_t *b, uint32_t p)
{
    return (uint32_t)b[p] |
           ((uint32_t)b[p + 1] << 8) |
           ((uint32_t)b[p + 2] << 16) |
           ((uint32_t)b[p + 3] << 24);
}

uint64_t orknn_fb_u64(const uint8_t *b, uint32_t p)
{
    return (uint64_t)orknn_fb_u32(b, p) |
           ((uint64_t)orknn_fb_u32(b, p + 4) << 32);
}

int32_t orknn_fb_i32(const uint8_t *b, uint32_t p)
{
    return (int32_t)orknn_fb_u32(b, p);
}

/* Follow an offset: pos + value_at(pos) */
uint32_t orknn_fb_follow(const uint8_t *b, uint32_t p)
{
    return p + orknn_fb_u32(b, p);
}

/* ======================================================================
 * Table field access
 * ====================================================================== */

/* Get field offset within a table. Returns absolute position, or 0 if absent. */
uint32_t orknn_fb_field(const uint8_t *b, uint32_t table, int field)
{
    uint32_t vt = table - orknn_fb_i32(b, table); /* vtable location */
    uint16_t vt_size = orknn_fb_u16(b, vt);
    uint32_t fo = 4 + field * 2; /* field offset within vtable */
    if (fo + 2 > vt_size)
        return 0;
    uint16_t off = orknn_fb_u16(b, vt + fo);
    return off ? table + off : 0;
}

/* ======================================================================
 * String / vector helpers
 * ====================================================================== */

/* Read a FlatBuffer string at field position fpos into out_str.
 * Returns string length, or 0 on failure. */
uint32_t orknn_fb_string(const uint8_t *b, uint32_t fpos,
                         char *out_str, uint32_t max_len)
{
    uint32_t str = orknn_fb_follow(b, fpos);
    uint32_t len = orknn_fb_u32(b, str);
    if (len == 0 || len >= max_len) {
        if (max_len > 0) out_str[0] = '\0';
        return 0;
    }
    memcpy(out_str, b + str + 4, len);
    out_str[len] = '\0';
    return len;
}

/* Get vector length at field position fpos */
uint32_t orknn_fb_vec_len(const uint8_t *b, uint32_t fpos)
{
    uint32_t vec = orknn_fb_follow(b, fpos);
    return orknn_fb_u32(b, vec);
}

/* Follow vector[index] offset (for vectors of tables/strings) */
uint32_t orknn_fb_vec_at(const uint8_t *b, uint32_t fpos, uint32_t index)
{
    uint32_t vec = orknn_fb_follow(b, fpos);
    uint32_t count = orknn_fb_u32(b, vec);
    if (index >= count)
        return 0;
    uint32_t elem_pos = vec + 4 + index * 4;
    return orknn_fb_follow(b, elem_pos);
}

/* Read a byte vector: returns pointer + length */
const uint8_t *orknn_fb_bytes(const uint8_t *b, uint32_t fpos, uint32_t *len)
{
    uint32_t vec = orknn_fb_follow(b, fpos);
    *len = orknn_fb_u32(b, vec);
    return b + vec + 4;
}

/* Read a scalar byte at field position */
uint8_t orknn_fb_byte(const uint8_t *b, uint32_t fpos)
{
    return b[fpos];
}
