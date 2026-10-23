#include "text/unicode.h"

#include <utf8proc/utf8proc.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <stdexcept>

namespace ninfer::text::unicode_internal {
namespace {

bool is_ascii_whitespace(std::int32_t codepoint) noexcept {
    return codepoint == ' ' || codepoint == '\t' || codepoint == '\n' || codepoint == '\r' ||
           codepoint == '\v' || codepoint == '\f';
}

std::string replace_invalid_utf8(std::string_view text) {
    std::string sanitized;
    sanitized.reserve(text.size());
    for (std::size_t offset = 0; offset < text.size();) {
        utf8proc_int32_t codepoint = 0;
        const utf8proc_ssize_t length = utf8proc_iterate(
            reinterpret_cast<const utf8proc_uint8_t*>(text.data() + offset),
            static_cast<utf8proc_ssize_t>(text.size() - offset), &codepoint);
        if (length < 0) {
            sanitized.append("\xEF\xBF\xBD");
            ++offset;
        } else {
            sanitized.append(text.data() + offset, static_cast<std::size_t>(length));
            offset += static_cast<std::size_t>(length);
        }
    }
    return sanitized;
}

} // namespace

std::string normalize_nfc(std::string_view text) {
    // Every byte below 0x80 is a starter with combining class zero and takes part in no canonical
    // composition, so pure ASCII is already in NFC and utf8proc has nothing to do with it. The
    // scan is one pass the compiler vectorises against utf8proc's decode, map and recompose.
    bool ascii_only = true;
    for (const char byte : text) {
        if (static_cast<unsigned char>(byte) >= 0x80u) {
            ascii_only = false;
            break;
        }
    }
    if (ascii_only) { return std::string(text); }

    const auto normalize = [](std::string_view input) -> std::optional<std::string> {
        utf8proc_uint8_t* mapped = nullptr;
        const utf8proc_ssize_t result =
            utf8proc_map(reinterpret_cast<const utf8proc_uint8_t*>(input.data()),
                         static_cast<utf8proc_ssize_t>(input.size()), &mapped,
                         static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
        if (result < 0) {
            std::free(mapped);
            return std::nullopt;
        }
        std::string normalized(reinterpret_cast<const char*>(mapped),
                               static_cast<std::size_t>(result));
        std::free(mapped);
        return normalized;
    };

    if (const auto normalized = normalize(text)) { return *normalized; }

    // A client can replay a message whose text contains a partial or otherwise
    // malformed UTF-8 sequence. Rejecting the whole conversation at NFC time
    // makes the session unrecoverable, so replace the malformed bytes with the
    // standard replacement character and keep the rest of the prompt intact.
    const std::string sanitized = replace_invalid_utf8(text);
    if (sanitized != text) {
        if (const auto normalized = normalize(sanitized)) {
            static std::atomic<bool> warned{false};
            if (!warned.exchange(true)) {
                std::fprintf(stderr,
                             "NINFER invalid UTF-8 in prompt text replaced with U+FFFD\n");
            }
            return *normalized;
        }
    }
    throw std::invalid_argument("failed to normalize UTF-8 text as NFC: invalid UTF-8");
}

CodepointSpan utf8_codepoint_at(std::string_view text, std::size_t offset,
                                std::string_view context) {
    if (offset >= text.size()) {
        throw std::out_of_range("UTF-8 codepoint offset exceeds " + std::string(context));
    }
    utf8proc_int32_t codepoint = 0;
    const utf8proc_ssize_t length =
        utf8proc_iterate(reinterpret_cast<const utf8proc_uint8_t*>(text.data() + offset),
                         static_cast<utf8proc_ssize_t>(text.size() - offset), &codepoint);
    if (length < 0) {
        throw std::invalid_argument("invalid UTF-8 in " + std::string(context) + ": " +
                                    utf8proc_errmsg(length));
    }
    if (length == 0) {
        throw std::invalid_argument("truncated UTF-8 sequence in " + std::string(context));
    }
    return CodepointSpan{codepoint, offset, static_cast<std::size_t>(length)};
}

std::vector<CodepointSpan> utf8_codepoints(std::string_view text, std::string_view context) {
    std::vector<CodepointSpan> codepoints;
    for (std::size_t offset = 0; offset < text.size();) {
        const CodepointSpan codepoint = utf8_codepoint_at(text, offset, context);
        codepoints.push_back(codepoint);
        offset += codepoint.length;
    }
    return codepoints;
}

std::string codepoint_to_utf8(std::int32_t codepoint) {
    utf8proc_uint8_t buffer[4];
    const utf8proc_ssize_t length = utf8proc_encode_char(codepoint, buffer);
    if (length <= 0) { throw std::invalid_argument("invalid Unicode codepoint"); }
    return {reinterpret_cast<const char*>(buffer), static_cast<std::size_t>(length)};
}

bool is_letter(std::int32_t codepoint) noexcept {
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_LU:
    case UTF8PROC_CATEGORY_LL:
    case UTF8PROC_CATEGORY_LT:
    case UTF8PROC_CATEGORY_LM:
    case UTF8PROC_CATEGORY_LO:
        return true;
    default:
        return false;
    }
}

bool is_mark(std::int32_t codepoint) noexcept {
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_MN:
    case UTF8PROC_CATEGORY_MC:
    case UTF8PROC_CATEGORY_ME:
        return true;
    default:
        return false;
    }
}

bool is_number(std::int32_t codepoint) noexcept {
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_ND:
    case UTF8PROC_CATEGORY_NL:
    case UTF8PROC_CATEGORY_NO:
        return true;
    default:
        return false;
    }
}

bool is_whitespace(std::int32_t codepoint) noexcept {
    if (is_ascii_whitespace(codepoint)) { return true; }
    switch (utf8proc_category(codepoint)) {
    case UTF8PROC_CATEGORY_ZS:
    case UTF8PROC_CATEGORY_ZL:
    case UTF8PROC_CATEGORY_ZP:
        return true;
    default:
        return false;
    }
}

} // namespace ninfer::text::unicode_internal
