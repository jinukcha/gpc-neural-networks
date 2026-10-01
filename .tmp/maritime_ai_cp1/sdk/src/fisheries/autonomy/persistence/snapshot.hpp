#pragma once

#include "../contracts/state.hpp"

#include <optional>
#include <string>
#include <string_view>

namespace maritime::fisheries::autonomy::detail {

std::string encode_snapshot(
    const RuntimeState& state,
    std::string_view mission_snapshot);
std::optional<RestoredState> decode_snapshot(
    std::string_view payload,
    const SkipperLimits& limits);

}  // namespace maritime::fisheries::autonomy::detail
