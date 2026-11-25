#ifndef __STAGING_POOL_HPP
#define __STAGING_POOL_HPP

#include "buffer.hpp"

#include <vulkan/vulkan.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <unordered_map>
#include <vector>

struct device;

constexpr VkDeviceSize kStagingPoolMaxIdleBytes = 128ull << 20;
constexpr auto kStagingPoolIdleTimeout = std::chrono::seconds(5);

// A buffer only enters the pool once the GPU has finished with it, so taking one needs no
// barrier. It's called from several threads, hence the mutex.
class StagingBufferPool {
public:
    using Clock = std::chrono::steady_clock;

    StagingBufferPool(struct device *dev, VkDeviceSize maxIdleBytes, Clock::duration idleTimeout);
    ~StagingBufferPool();

    StagingBufferPool(const StagingBufferPool&) = delete;
    StagingBufferPool& operator=(const StagingBufferPool&) = delete;

    std::unique_ptr<struct buffer> Acquire(VkDeviceSize capacity);
    void ReturnAll(std::vector<std::unique_ptr<struct buffer>> buffers);
    // Returns when the next idle buffer expires, or nothing if the pool is empty.
    std::optional<Clock::time_point> Trim();
    // Returns the bytes it freed. Must be called before the VkDevice is destroyed.
    VkDeviceSize Clear();
    void LogStats();

private:
    struct Entry {
        std::unique_ptr<struct buffer> buf;
        Clock::time_point idleSince;
    };
    using EntryList = std::list<Entry>;

    std::unique_ptr<struct buffer> PopOldestLocked();
    void Destroy(std::vector<std::unique_ptr<struct buffer>>& doomed);

    struct device *dev;
    const VkDeviceSize maxIdleBytes;
    const Clock::duration idleTimeout;

    std::mutex lock;
    // Every idle buffer, longest-idle first. Returned buffers are appended to the back, so the
    // list stays sorted.
    EntryList idle;
    // Longest-idle first, like `idle`: Acquire() takes from the back, eviction and Trim() from the front.
    std::unordered_map<VkDeviceSize, std::deque<EntryList::iterator>> idleByCapacity;
    VkDeviceSize idleBytes = 0;
    uint64_t hits = 0;
    uint64_t misses = 0;
    uint64_t returned = 0;
    std::atomic<uint64_t> destroyed {0};
};

#endif
