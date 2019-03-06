#version 450
#extension GL_ARB_separate_shader_objects : enable

layout(binding = 0) uniform Camera 
{
	mat4 M; // Model
	mat4 V; // View
	mat4 P; // Projection
} cam;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inTex;
layout(location = 2) in vec3 inColor;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTex;

void main() {
    gl_Position = cam.P * cam.V * cam.M * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;
	fragTex = inTex;
}
