#pragma once

#include "../contracts/state.hpp"

namespace maritime::fisheries::autonomy::detail {

bool validate_profile(const SkipperProfile& profile);
bool validate_ground(const FishingGroundProfile& ground);
bool validate_observation(const FishingGroundObservation& observation);

bool initialize_grounds(
    RuntimeState& state,
    std::vector<FishingGroundProfile> grounds,
    const SkipperLimits& limits);
bool update_belief(
    RuntimeState& state,
    const FishingGroundObservation& observation,
    const SkipperLimits& limits);
void age_beliefs(RuntimeState& state, Tick tick);
std::vector<FishingGroundBelief> belief_views(const RuntimeState& state);

}  // namespace maritime::fisheries::autonomy::detail
