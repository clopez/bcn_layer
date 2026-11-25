#ifndef __BUFFER_HPP
#define __BUFFER_HPP
#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string_view>

struct device;

// Available does not mean the GPU has finished; reuse still needs a barrier
// unless one has already been recorded since the buffer was released.
enum class StagingReuseState {
    NotReusable,
    Available,
    Reserved,
};

struct buffer {
    VkBuffer handle = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    // Capacity of the buffer. Reused staging buffers are rounded up to a power of two, so this
    // can be larger than what the current conversion needs.
    VkDeviceSize size = 0;
    // Bytes used by the current conversion, including format alignment. 0 for game-owned buffers.
    VkDeviceSize usedSize = 0;
    VkDeviceSize offset = 0;
    struct device *device = nullptr;
    const VkAllocationCallbacks *alloc = nullptr;
    VkFormat format = VK_FORMAT_UNDEFINED;
    std::string_view label;
    int id = 0;
    int width = 0;
    int height = 0;
    StagingReuseState stagingReuseState = StagingReuseState::NotReusable;
    // Value of StagingResources::reuseBarrierEpoch when this buffer became Available. If a reuse
    // barrier has been recorded since then, the buffer can be reused without another one.
    uint64_t releaseEpoch = 0;

    // Covers the bytes used by the current conversion, or the whole buffer when usedSize is 0.
    VkDescriptorBufferInfo descriptorInfo() const
    {
        return { handle, 0, usedSize ? usedSize : VK_WHOLE_SIZE };
    }
    // Same, starting at an offset that comes from the game.
    VkDescriptorBufferInfo descriptorInfoAt(VkDeviceSize absoluteOffset) const;
};

struct buffer *find_buffer(VkBuffer);
std::unique_ptr<struct buffer> create_staging_buffer(struct device *dev, VkDeviceSize size, VkFormat format, int width, int height, std::string_view label = "");

constexpr VkDeviceSize AlignUp(VkDeviceSize value, VkDeviceSize alignment)
{
    return (value + alignment - 1) / alignment * alignment;
}

VkDeviceSize staging_buffer_aligned_size(VkFormat format, VkDeviceSize bytes);

#endif
