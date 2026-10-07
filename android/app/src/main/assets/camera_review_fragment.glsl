precision mediump float;

uniform sampler2D u_texture;
uniform float u_exposure;
uniform float u_contrast;

varying vec2 v_texcoord;

void main() {
    vec3 color = texture2D(u_texture, v_texcoord).rgb;

    float expMul = pow(2.0, u_exposure);
    float cFactor = tan((u_contrast + 1.0) * 0.78539816339);

    color = color * expMul;
    color = (color - 0.5) * cFactor + 0.5;
    color = clamp(color, 0.0, 1.0);

    gl_FragColor = vec4(color, 1.0);
}
