precision highp float;

uniform sampler2D u_texture;
uniform vec3 u_mu_lin;
uniform mat3 u_W_lin;
uniform mat3 u_Winv_lin;
uniform mat3 u_rotation;
uniform float u_push_factor;
uniform float u_var_restore;
uniform int u_channel_mode;
uniform int u_split_active;
uniform float u_split_pos;
uniform float u_split_line_w;

varying vec2 v_texcoord;

// sRGB to Linear conversion — vectorized, no dynamic indexing (GLSL ES 1.0 safe)
vec3 srgb_to_linear(vec3 c) {
    vec3 a = vec3(0.055);
    return mix(c / 12.92, pow((c + a) / 1.055, vec3(2.4)), step(vec3(0.04045), c));
}

// Linear to sRGB conversion — vectorized, no dynamic indexing (GLSL ES 1.0 safe)
vec3 linear_to_srgb(vec3 c) {
    vec3 a = vec3(0.055);
    return mix(c * 12.92, 1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - a, step(vec3(0.0031308), c));
}

void main() {
    // Step 1: Sample the sRGB image
    vec3 color_srgb = texture2D(u_texture, v_texcoord).rgb;

    vec3 base_color;
    if (u_split_active == 1 && v_texcoord.x < u_split_pos) {
        base_color = color_srgb;
    } else {
        // Step 2: Convert sRGB to linear RGB
        // Python equivalent: img_lin = ColorUtils.srgb_to_linear_np(img_srgb)
        vec3 color_linear = srgb_to_linear(color_srgb);

        // Step 3: Subtract mean (center the data)
        // Python equivalent: Xl - mu_lin
        vec3 centered = color_linear - u_mu_lin;

        // Step 4: Apply whitening matrix (ZCA whitening)
        // Python equivalent: z = (Xl - mu_lin) @ W_lin.T
        // Note: In GLSL, matrix multiplication is mat * vec, which is equivalent to vec @ mat.T in Python
        vec3 whitened = u_W_lin * centered;

        // Step 5: Apply rotation matrix
        // Python equivalent: z2 = Ruser * z
        vec3 rotated = u_rotation * whitened;

        // Step 5b: Apply push factor (scale the rotated vector)
        vec3 pushed = rotated * u_push_factor;

        // Step 6: Inverse whitening (de-whitening) with variance restore control
        // eff_Winv = I + u_var_restore * (Winv - I)
        // u_var_restore=1.0 → full Winv (full variance restore)
        // u_var_restore=0.0 → identity (no variance restore, stays whitened)
        mat3 eff_Winv = mat3(1.0) + u_var_restore * (u_Winv_lin - mat3(1.0));
        vec3 dewhitened = eff_Winv * pushed;

        // Step 7: Add mean back
        vec3 result_linear = u_mu_lin + dewhitened;

        // Step 8: Clamp to valid range
        result_linear = clamp(result_linear, 0.0, 1.0);

        // Step 9: Convert back to sRGB for display
        // Python equivalent: img_srgb = ColorUtils.linear_to_srgb_np(img_linear)
        base_color = linear_to_srgb(result_linear);
    }

    // Step 10: Apply channel mode if needed
    // 0 = COLOR (full color), 1 = RED only, 2 = GREEN only, 3 = BLUE only
    vec3 final_color = base_color;
    if (u_split_active == 0 || v_texcoord.x >= u_split_pos) {
        if (u_channel_mode == 1) {
            final_color = vec3(base_color.r, base_color.r, base_color.r);
        } else if (u_channel_mode == 2) {
            final_color = vec3(base_color.g, base_color.g, base_color.g);
        } else if (u_channel_mode == 3) {
            final_color = vec3(base_color.b, base_color.b, base_color.b);
        }
    }

    if (u_split_active == 1 && abs(v_texcoord.x - u_split_pos) < u_split_line_w) {
        final_color = vec3(1.0);
    }

    gl_FragColor = vec4(final_color, 1.0);
}
