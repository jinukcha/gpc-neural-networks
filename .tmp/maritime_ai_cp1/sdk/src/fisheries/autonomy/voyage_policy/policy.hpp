#pragma once

#include "../contracts/state.hpp"

namespace maritime::fisheries::autonomy::detail {

PolicyResult evaluate_policy(
    const RuntimeState& state,
    const VoyageFacts& facts);

}  // namespace maritime::fisheries::autonomy::detail
