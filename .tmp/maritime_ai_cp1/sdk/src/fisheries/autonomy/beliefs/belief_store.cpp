#include "belief_store.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace maritime::fisheries::autonomy::detail {
namespace {

bool finite_ratio(const double value) {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

void age_one(
    GroundState& ground,
    const Tick tick,
    const double uncertainty_growth_per_tick) {
    if (tick <= ground.belief.last_observed_tick) return;
    const auto elapsed =
        static_cast<double>(tick - ground.belief.last_observed_tick);
    ground.belief.uncertainty_ratio = std::clamp(
        ground.belief.uncertainty_ratio
            + elapsed * uncertainty_growth_per_tick,
        0.0, 1.0);
}

}  // namespace

bool validate_profile(const SkipperProfile& profile) {
    return !profile.profile_id.empty()
        && std::isfinite(profile.exploration_weight)
        && profile.exploration_weight >= 0.0
        && finite_ratio(profile.uncertainty_growth_per_tick)
        && std::isfinite(profile.travel_cost_weight)
        && profile.travel_cost_weight >= 0.0
        && std::isfinite(profile.minimum_viable_catch_rate_kg_per_hour)
        && profile.minimum_viable_catch_rate_kg_per_hour >= 0.0
        && profile.minimum_quota_remaining_kg >= 0.0
        && finite_ratio(profile.maximum_bycatch_ratio)
        && finite_ratio(profile.maximum_hold_load_ratio)
        && finite_ratio(profile.minimum_quality_ratio)
        && finite_ratio(profile.maximum_processing_backlog_ratio)
        && profile.maximum_sets > 0U;
}

bool validate_ground(const FishingGroundProfile& ground) {
    return !ground.key.empty()
        && ground.key.find('|') == std::string::npos
        && std::isfinite(ground.north_m)
        && std::isfinite(ground.east_m)
        && std::isfinite(ground.travel_cost_credits)
        && ground.travel_cost_credits >= 0.0
        && std::isfinite(ground.prior_catch_rate_kg_per_hour)
        && ground.prior_catch_rate_kg_per_hour >= 0.0
        && finite_ratio(ground.prior_bycatch_ratio)
        && std::isfinite(ground.prior_market_value_per_kg)
        && ground.prior_market_value_per_kg >= 0.0;
}

bool validate_observation(const FishingGroundObservation& observation) {
    return !observation.ground_key.empty()
        && std::isfinite(observation.catch_rate_kg_per_hour)
        && observation.catch_rate_kg_per_hour >= 0.0
        && finite_ratio(observation.bycatch_ratio)
        && std::isfinite(observation.market_value_per_kg)
        && observation.market_value_per_kg >= 0.0
        && finite_ratio(observation.quality_risk_ratio)
        && finite_ratio(observation.hold_risk_ratio);
}

bool initialize_grounds(
    RuntimeState& state,
    std::vector<FishingGroundProfile> grounds,
    const SkipperLimits& limits) {
    if (grounds.empty() || grounds.size() > limits.max_grounds) return false;
    std::map<std::string, GroundState, std::less<>> prepared;
    for (auto& profile : grounds) {
        if (!validate_ground(profile) || prepared.contains(profile.key)) {
            return false;
        }
        FishingGroundBelief belief;
        belief.ground_key = profile.key;
        belief.estimated_catch_rate_kg_per_hour =
            profile.prior_catch_rate_kg_per_hour;
        belief.estimated_bycatch_ratio = profile.prior_bycatch_ratio;
        belief.estimated_market_value_per_kg =
            profile.prior_market_value_per_kg;
        belief.access_allowed = profile.access_allowed;
        prepared.emplace(
            profile.key,
            GroundState{std::move(profile), std::move(belief), {}});
    }
    state.grounds = std::move(prepared);
    return true;
}

bool update_belief(
    RuntimeState& state,
    const FishingGroundObservation& observation,
    const SkipperLimits& limits) {
    if (!validate_observation(observation)) return false;
    const auto found = state.grounds.find(observation.ground_key);
    if (found == state.grounds.end()) return false;
    auto& ground = found->second;
    age_one(ground, observation.tick,
            state.profile.uncertainty_growth_per_tick);
    if (ground.observations.size() >= limits.max_observations_per_ground) {
        ground.observations.pop_front();
    }
    ground.observations.push_back(observation);
    const auto retained = static_cast<double>(ground.observations.size());
    const auto weight = 1.0 / retained;
    auto& belief = ground.belief;
    belief.estimated_catch_rate_kg_per_hour += weight
        * (observation.catch_rate_kg_per_hour
           - belief.estimated_catch_rate_kg_per_hour);
    belief.estimated_bycatch_ratio += weight
        * (observation.bycatch_ratio - belief.estimated_bycatch_ratio);
    belief.estimated_market_value_per_kg += weight
        * (observation.market_value_per_kg
           - belief.estimated_market_value_per_kg);
    belief.quality_risk_ratio += weight
        * (observation.quality_risk_ratio - belief.quality_risk_ratio);
    belief.hold_risk_ratio += weight
        * (observation.hold_risk_ratio - belief.hold_risk_ratio);
    belief.last_observed_tick = observation.tick;
    belief.observation_count = static_cast<std::uint32_t>(
        std::min<std::size_t>(ground.observations.size(),
                              limits.max_observations_per_ground));
    belief.uncertainty_ratio = std::clamp(
        1.0 / std::sqrt(retained + 1.0), 0.05, 1.0);
    belief.access_allowed = observation.access_allowed;
    belief.closed = observation.closed;
    ++belief.revision;
    return true;
}

void age_beliefs(RuntimeState& state, const Tick tick) {
    for (auto& entry : state.grounds) {
        age_one(entry.second, tick,
                state.profile.uncertainty_growth_per_tick);
    }
}

std::vector<FishingGroundBelief> belief_views(const RuntimeState& state) {
    std::vector<FishingGroundBelief> result;
    result.reserve(state.grounds.size());
    for (const auto& entry : state.grounds) {
        result.push_back(entry.second.belief);
    }
    return result;
}

}  // namespace maritime::fisheries::autonomy::detail
