__constant int offsetx[4] = {1, -1, 0, 0};
__constant int offsety[4] = {0, 0, 1, -1};
__constant float4 solidities = {0.999, 0.5/800., 0.5, 0.0};  // red=solid, green=gas, blue=liquid
__constant float4 solidities_inv = {0.001, 1 - 0.5/800., 0.5, 1.0};  // 1 - solidities
__constant float4 solidities_inv2 = {1./0.999, 1.0 / (0.5/800.), 1.0/0.5, INFINITY};  // 1/solidities

float vmax(float4 v) {
    return max(max(v.x, v.y), v.z);  // v.w not included
}
float vsum(float4 v) {
    return v.x + v.y + v.z;
}

float4 compute_gravity(float4 north, float4 south) {
    // Gravity from north to south
    return clamp(((float4)(1,1,1,1) - south) * solidities - vmax(south * solidities), 0, north);
    // return clamp(north + solidities_inv * (vsum(north * solidities) - vsum(south * solidities)), 0, north);
    // return clamp(north - solidities_inv2 * vmax(south * solidities), 0, north);
    // return 0.0f;


    // Cool looking:
    // return clamp(((float4)(1,1,1,1) - south) - solidities_inv * vsum(south * solidities), 0, north);
}

float4 compute_diffusion(float4 pixel, float4 neighbor) {
    // Diffusion from neighbor to pixel
    return 0.2f * clamp((neighbor - pixel) * solidities_inv + (vmax(neighbor) - vmax(pixel)), -pixel, neighbor);
    // return 0.2f * clamp(max(neighbor - pixel, (float4)(0,0,0,0)) * ((float4)(1,1,1,1) - solidities) * (vsum(neighbor) - vsum(pixel)), -pixel, neighbor);
    // return 0.2f * clamp((neighbor - pixel + (vmax(neighbor) - vmax(pixel))) * solidities_inv, -pixel, neighbor);
    // return 0.2f * clamp((neighbor - pixel) * solidities_inv + (vsum(neighbor * solidities) - vsum(pixel * solidities)) * solidities_inv, -pixel, neighbor);
    // return 0.2f * clamp((vmax(neighbor * solidities) - vmax(pixel * solidities)) * solidities_inv, -pixel, neighbor);
    // return 0.f;


    // Cool looking:
    // return 0.2f * clamp(((float4)(1,1,1,1) - pixel) * solidities_inv - (vsum(neighbor * solidities) - vsum(pixel * solidities)) * solidities_inv, -pixel, neighbor);
    // return 0.2f * clamp(((float4)(1,1,1,1) - pixel) * solidities_inv - (vsum(neighbor * solidities) - vsum(pixel * solidities)) * solidities_inv, 0, neighbor);
}

__kernel void apply_physics(__global const float4 *input, __global float4 *output, const unsigned int width, const unsigned int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x < (int)width && y < (int)height) {
        int index = y * width + x;
        float4 pixel = input[index];
        float4 out = pixel;

        // gravity
        if (y - 1 >= 0) { // gravity from above
            int north_index = (y-1) * width + x;
            float4 north = input[north_index];
            float4 gravity = compute_gravity(north, pixel);
            out += gravity;
        }
        if (y + 1 < (int)height) {  // gravity to below
            int south_index = (y+1) * width + x;
            float4 south = input[south_index];
            float4 gravity = compute_gravity(pixel, south);
            out -= gravity;
        }

        // diffusion
        for (int i = 0; i < 4; i++) {
            int nx = x + offsetx[i];
            int ny = y + offsety[i];
            if (nx >= 0 && nx < (int)width && ny >= 0 && ny < (int)height) {
                int neighbor_index = ny * width + nx;
                float4 neighbor = input[neighbor_index];
                out += compute_diffusion(pixel, neighbor);
            }
        }

        output[index] = out;
    }
}
