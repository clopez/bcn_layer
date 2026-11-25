#include "staging_pool.hpp"

#include "bcn_layer.hpp"
#include "logger.hpp"

StagingBufferPool::StagingBufferPool(struct device *dev, VkDeviceSize maxIdleBytes, Clock::duration idleTimeout)
    : dev(dev), maxIdleBytes(maxIdleBytes), idleTimeout(idleTimeout)
{
}

StagingBufferPool::~StagingBufferPool()
{
    Clear();
}

std::unique_ptr<struct buffer> StagingBufferPool::Acquire(VkDeviceSize capacity)
{
    std::lock_guard<std::mutex> guard(lock);
    auto it = idleByCapacity.find(capacity);
    if (it == idleByCapacity.end() || it->second.empty()) {
        misses++;
        return nullptr;
    }

    // Take the most recently returned buffer; the oldest ones are left for Trim().
    EntryList::iterator entry = it->second.back();
    it->second.pop_back();
    std::unique_ptr<struct buffer> buf = std::move(entry->buf);
    idle.erase(entry);
    idleBytes -= buf->size;
    hits++;
    return buf;
}

std::unique_ptr<struct buffer> StagingBufferPool::PopOldestLocked()
{
    Entry &oldest = idle.front();
    std::unique_ptr<struct buffer> buf = std::move(oldest.buf);
    // The oldest entry overall is also the oldest one of its capacity.
    idleByCapacity[buf->size].pop_front();
    idle.pop_front();
    idleBytes -= buf->size;
    return buf;
}

void StagingBufferPool::ReturnAll(std::vector<std::unique_ptr<struct buffer>> buffers)
{
    std::vector<std::unique_ptr<struct buffer>> doomed;
    {
        std::lock_guard<std::mutex> guard(lock);
        const Clock::time_point now = Clock::now();
        for (auto &buf : buffers) {
            if (!buf)
                continue;
            returned++;
            if (buf->size > maxIdleBytes) {
                doomed.push_back(std::move(buf));
                continue;
            }
            buf->stagingReuseState = StagingReuseState::NotReusable;
            idleBytes += buf->size;
            const VkDeviceSize capacity = buf->size;
            idle.push_back({ std::move(buf), now });
            idleByCapacity[capacity].push_back(std::prev(idle.end()));
        }

        while (idleBytes > maxIdleBytes && !idle.empty())
            doomed.push_back(PopOldestLocked());
    }
    Destroy(doomed);
}

std::optional<StagingBufferPool::Clock::time_point> StagingBufferPool::Trim()
{
    std::vector<std::unique_ptr<struct buffer>> doomed;
    std::optional<Clock::time_point> nextExpiry;
    {
        std::lock_guard<std::mutex> guard(lock);
        const Clock::time_point cutoff = Clock::now() - idleTimeout;
        while (!idle.empty() && idle.front().idleSince < cutoff)
            doomed.push_back(PopOldestLocked());
        if (!idle.empty())
            nextExpiry = idle.front().idleSince + idleTimeout;
    }
    if (!doomed.empty() && dev->profile_transfers)
        Logger::log("info", "staging pool: trimmed %zu idle buffers", doomed.size());
    Destroy(doomed);
    return nextExpiry;
}

VkDeviceSize StagingBufferPool::Clear()
{
    std::vector<std::unique_ptr<struct buffer>> doomed;
    VkDeviceSize freedBytes;
    {
        std::lock_guard<std::mutex> guard(lock);
        for (auto &entry : idle)
            doomed.push_back(std::move(entry.buf));
        idle.clear();
        idleByCapacity.clear();
        freedBytes = idleBytes;
        idleBytes = 0;
    }
    Destroy(doomed);
    return freedBytes;
}

void StagingBufferPool::LogStats()
{
    std::lock_guard<std::mutex> guard(lock);
    Logger::log("info", "staging pool: %.1f MiB idle, %llu hits, %llu misses, %llu returned, %llu destroyed",
        idleBytes / (1024.0 * 1024.0), (unsigned long long) hits, (unsigned long long) misses,
        (unsigned long long) returned, (unsigned long long) destroyed.load());
}

void StagingBufferPool::Destroy(std::vector<std::unique_ptr<struct buffer>>& doomed)
{
    for (auto& buf : doomed) {
        if (!buf)
            continue;
        dev->table.DestroyBuffer(dev->handle, buf->handle, buf->alloc);
        dev->table.FreeMemory(dev->handle, buf->memory, buf->alloc);
    }
    destroyed += doomed.size();
    doomed.clear();
}
