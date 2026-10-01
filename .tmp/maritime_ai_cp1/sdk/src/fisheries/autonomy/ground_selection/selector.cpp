#include "selector.hpp"

#include <cmath>
#include <cstdint>
#include <limits>

namespace maritime::fisheries::autonomy::detail {
namespace {

std::uint64_t stable_hash(const std::uint64_t seed, std::string_view key) {
    std::uint64_t value = 1469598103934665603ULL ^ seed;
    for (const char raw_character : key) {
        const auto character = static_cast<unsigned char>(raw_character);
        value ^= static_cast<std::uint64_t>(character);
        value *= 1099511628211ULL;
    }
    return value;
}

DecisionWhy score_ground(
    const RuntimeState& state,
    const GroundState& ground,
    const VoyageFacts& facts) {
    const auto& profile = state.profile;
    const auto& belief = ground.belief;
    DecisionWhy why;
    why.expected_value = belief.estimated_catch_rate_kg_per_hour
        * belief.estimated_market_value_per_kg;
    why.travel_penalty = ground.profile.travel_cost_credits
        * profile.travel_cost_weight;
    why.bycatch_penalty = belief.estimated_bycatch_ratio
        * profile.bycatch_risk_weight;
    why.quality_penalty = (belief.quality_risk_ratio
        + (1.0 - facts.weighted_quality_ratio))
        * profile.quality_risk_weight;
    why.hold_penalty = (belief.hold_risk_ratio + facts.hold_load_ratio)
        * profile.hold_risk_weight;
    why.information_bonus = belief.uncertainty_ratio
        * profile.exploration_weight
        * (1.0 + belief.estimated_market_value_per_kg);
    const auto tie = static_cast<double>(
        stable_hash(state.seed, ground.profile.key) % 1000000ULL)
        / 1000000.0;
    why.total_utility = why.expected_value
        - why.travel_penalty
        - why.bycatch_penalty
        - why.quality_penalty
        - why.hold_penalty
        + why.information_bonus
        + tie * 1.0e-9;
    why.reason = "highest_admitted_ground_utility";
    return why;
}

}  // namespace

std::optional<GroundScore> select_ground(
    const RuntimeState& state,
    const VoyageFacts& facts,
    const std::string_view excluded_ground) {
    std::optional<GroundScore> best;
    double best_utility = -std::numeric_limits<double>::infinity();
    for (const auto& entry : state.grounds) {
        const auto& ground = entry.second;
        if (ground.profile.key == excluded_ground
            || !ground.profile.access_allowed
            || !ground.belief.access_allowed
            || ground.belief.closed) {
            continue;
        }
        auto why = score_ground(state, ground, facts);
        if (why.total_utility <= best_utility) continue;
        best_utility = why.total_utility;
        best = GroundScore{ground.profile.key, std::move(why)};
    }
    return best;
}

}  // namespace maritime::fisheries::autonomy::detail
