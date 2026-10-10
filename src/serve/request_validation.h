#pragma once

#include "serve/request.h"
#include "serve/request_json.h"

#include <cstddef>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::serve {

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {});

// Settles every parsed image detail: a value outside auto, low and high is refused with
// image_detail_not_supported (`field`, under request parameter `param`) unless the server reads such
// values as auto (--lenient-image-detail). Low selects the bounded Vision profile.
void settle_image_details(std::vector<ChatTurn>& turns, const RequestLimits& limits,
                          const std::string& param, const std::string& field);

std::optional<int> optional_int(const RequestJson& object, const char* key);
std::optional<double> optional_number(const RequestJson& object, const char* key);
bool optional_bool(const RequestJson& object, const char* key, bool fallback);

[[nodiscard]] bool valid_tool_name(std::string_view name, std::size_t maximum_length) noexcept;

} // namespace ninfer::serve
