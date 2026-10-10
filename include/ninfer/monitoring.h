#pragma once
#include <cstdint>
namespace ninfer {
// Published execution-lane snapshot. Retained KVMem history is not a saved slot.
struct ExecutionSlotState {
    std::uint32_t id = 0;
    bool processing = false;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t cached_tokens = 0;
    std::uint32_t generated_tokens = 0;
};
} // namespace ninfer
