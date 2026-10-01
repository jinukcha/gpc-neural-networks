#pragma once

#include "../contracts/state.hpp"

#include <optional>

namespace maritime::fisheries::autonomy::detail {

std::optional<GroundScore> select_ground(
    const RuntimeState& state,
    const VoyageFacts& facts,
    std::string_view excluded_ground = {});

}  // namespace maritime::fisheries::autonomy::detail
