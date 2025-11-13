#shader compute
#version 460

layout(local_size_x = 8, local_size_y = 8, local_size_z = 8) in;

layout(set = 0, binding = 6, r32f) uniform image3D voxelRead[4]; // input
layout(set = 0, binding = 7, r32f) uniform image3D voxelWrite[4]; // output
layout(push_constant) uniform MipInfo {
    float level; //Level you are computing (should start on 1, not 0)
} mipinfo;

void main() {
    int mipLevel = int(mipinfo.level);
    int prevMipLevel = int(mipinfo.level) - 1;

    ivec3 id = ivec3(gl_GlobalInvocationID.xyz);

   
    // Size of the target mip level
    ivec3 size = imageSize(voxelRead[mipLevel]);
    if (any(greaterThanEqual(id, size)))
        return;

    ivec3 prevSize = imageSize(voxelRead[prevMipLevel]);

    ivec3 base = id * 2;
    float totalSum = 0.0;
    for(int x = 0; x < 2; ++x) {
        for(int y = 0; y < 2; ++y) {
            for(int z = 0; z < 2; ++z) {
                ivec3 c = base + ivec3(x,y,z);
                 if (any(greaterThanEqual(c, prevSize)) || any(lessThan(c, ivec3(0)))) 
                    // out of prev bounds -> treat as 0
                    continue;
                totalSum += imageLoad(voxelRead[prevMipLevel], c).r;
            }
        }
    }

    imageStore(voxelWrite[mipLevel], id, vec4(totalSum, 0.0, 0.0, 0.0));

}
