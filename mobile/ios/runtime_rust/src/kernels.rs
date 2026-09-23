#[inline(always)]
pub fn fp16_to_fp32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h & 0x7C00) >> 10) as u32;
    let mant = (h & 0x03FF) as u32;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let val = (mant as f32) * (1.0f32 / 16777216.0f32);
        if (h & 0x8000) != 0 { -val } else { val }
    } else if exp == 31 {
        if mant == 0 {
            f32::from_bits(sign | 0x7F800000)
        } else {
            f32::from_bits(sign | 0x7F800000 | (mant << 13))
        }
    } else {
        f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
    }
}

#[inline(always)]
pub fn fp32_to_fp16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;

    if exponent <= 0 {
        return sign;
    }
    if exponent >= 31 {
        return sign | 0x7c00;
    }

    sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16)
}

pub fn gemv_q4_0(
    weights: &[u8],
    input: &[f32],
    output: &mut [f32],
    k: usize,
    n: usize,
) {
    let blocks_per_row = k / 32;
    let block_size = 18;
    let row_stride = blocks_per_row * block_size;

    let w_base = weights.as_ptr();
    let in_base = input.as_ptr();
    let out_base = output.as_mut_ptr();

    for r in 0..n {
        let row_start = r * row_stride;
        let mut row_sum = 0.0f32;

        for b in 0..blocks_per_row {
            let b_start = row_start + b * block_size;
            let scale_bits = unsafe {
                u16::from_le_bytes([*w_base.add(b_start), *w_base.add(b_start + 1)])
            };
            let scale = fp16_to_fp32(scale_bits);
            let in_offset = b * 32;

            let block_qs = unsafe { w_base.add(b_start + 2) };
            let block_in = unsafe { in_base.add(in_offset) };

            let mut sum0 = 0.0f32;
            let mut sum1 = 0.0f32;

            for j in 0..16 {
                let byte_val = unsafe { *block_qs.add(j) };
                let q0 = ((byte_val & 0x0F) as i32) - 8;
                let q1 = ((byte_val >> 4) as i32) - 8;

                let x0 = unsafe { *block_in.add(j * 2) };
                let x1 = unsafe { *block_in.add(j * 2 + 1) };

                sum0 += (q0 as f32) * x0;
                sum1 += (q1 as f32) * x1;
            }
            row_sum += (sum0 + sum1) * scale;
        }

        unsafe {
            *out_base.add(r) = row_sum;
        }
    }
}

pub fn gemv_q4_0_swiglu(
    gate_weights: &[u8],
    up_weights: &[u8],
    input: &[f32],
    output: &mut [f32],
    k: usize,
    n: usize,
) {
    let blocks_per_row = k / 32;
    let row_stride = blocks_per_row * 18;

    for r in 0..n {
        let row_start = r * row_stride;
        let mut gate = 0.0f32;
        let mut up = 0.0f32;

        for b in 0..blocks_per_row {
            let block_start = row_start + b * 18;
            let gate_scale = fp16_to_fp32(u16::from_le_bytes([
                gate_weights[block_start],
                gate_weights[block_start + 1],
            ]));
            let up_scale = fp16_to_fp32(u16::from_le_bytes([
                up_weights[block_start],
                up_weights[block_start + 1],
            ]));
            let input_start = b * 32;
            let mut gate_block = 0.0f32;
            let mut up_block = 0.0f32;

            for j in 0..16 {
                let gate_byte = gate_weights[block_start + 2 + j];
                let up_byte = up_weights[block_start + 2 + j];
                let x0 = input[input_start + j * 2];
                let x1 = input[input_start + j * 2 + 1];

                gate_block += (((gate_byte & 0x0f) as i32 - 8) as f32) * x0
                    + (((gate_byte >> 4) as i32 - 8) as f32) * x1;
                up_block += (((up_byte & 0x0f) as i32 - 8) as f32) * x0
                    + (((up_byte >> 4) as i32 - 8) as f32) * x1;
            }

            gate += gate_block * gate_scale;
            up += up_block * up_scale;
        }

        output[r] = (gate / (1.0 + (-gate).exp())) * up;
    }
}

pub fn gemv_q8_0(
    weights: &[u8],
    input: &[f32],
    output: &mut [f32],
    k: usize,
    n: usize,
) {
    let blocks_per_row = k / 32;
    let block_size = 34;
    let row_stride = blocks_per_row * block_size;

    for r in 0..n {
        let row_start = r * row_stride;
        let mut row_sum = 0.0f32;

        for b in 0..blocks_per_row {
            let b_start = row_start + b * block_size;
            let scale_bits = u16::from_le_bytes([weights[b_start], weights[b_start + 1]]);
            let scale = fp16_to_fp32(scale_bits);
            let in_base = b * 32;

            let mut b_sum = 0.0f32;
            for j in 0..32 {
                let val = weights[b_start + 2 + j] as i8;
                b_sum += (val as f32) * input[in_base + j];
            }
            row_sum += b_sum * scale;
        }

        output[r] = row_sum;
    }
}

pub fn rms_norm(
    input: &[f32],
    weight_raw: &[u8],
    output: &mut [f32],
    dim: usize,
    eps: f32,
) {
    let mut sum_sq = 0.0f32;
    for i in 0..dim {
        sum_sq += input[i] * input[i];
    }
    let inv_std = 1.0f32 / ((sum_sq / (dim as f32)) + eps).sqrt();

    for i in 0..dim {
        let w_bits = u16::from_le_bytes([weight_raw[i * 2], weight_raw[i * 2 + 1]]);
        let w = fp16_to_fp32(w_bits);
        output[i] = input[i] * inv_std * w;
    }
}

pub fn apply_rope(
    vec: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    cos_table: &[f32],
    sin_table: &[f32],
    pos: usize,
) {
    let half_dim = head_dim / 2;
    let table_offset = pos * half_dim;
    for h in 0..num_heads {
        let head_offset = h * head_dim;
        for i in 0..half_dim {
            let cos_th = cos_table[table_offset + i];
            let sin_th = sin_table[table_offset + i];

            let v0 = vec[head_offset + i];
            let v1 = vec[head_offset + i + half_dim];

            vec[head_offset + i]            = v0 * cos_th - v1 * sin_th;
            vec[head_offset + i + half_dim] = v0 * sin_th + v1 * cos_th;
        }
    }
}
