#include "policy.hpp"

namespace maritime::fisheries::autonomy::detail {

PolicyResult evaluate_policy(
    const RuntimeState& state,
    const VoyageFacts& facts) {
    const auto& profile = state.profile;
    if (facts.gear_faulted) {
        return {true, false, false, "gear_fault_secure_and_abort"};
    }
    if (facts.quota_remaining_kg <= profile.minimum_quota_remaining_kg) {
        return {false, true, false, "quota_floor_reached"};
    }
    if (facts.current_bycatch_ratio >= profile.maximum_bycatch_ratio) {
        return {false, true, false, "bycatch_limit_reached"};
    }
    if (facts.hold_load_ratio >= profile.maximum_hold_load_ratio) {
        return {false, true, false, "hold_capacity_limit"};
    }
    if (facts.weighted_quality_ratio <= profile.minimum_quality_ratio) {
        return {false, true, false, "quality_floor_reached"};
    }
    if (facts.processing_backlog_ratio
        >= profile.maximum_processing_backlog_ratio) {
        return {false, true, false, "processing_backlog_limit"};
    }
    if (facts.sets_completed >= profile.maximum_sets) {
        return {false, true, false, "planned_sets_complete"};
    }
    if (facts.current_ground_closed || !facts.current_ground_accessible) {
        const bool can_relocate =
            state.relocation_count < profile.maximum_relocations;
        return {false, !can_relocate, can_relocate,
                can_relocate ? "ground_access_lost_relocate"
                             : "ground_access_lost_return"};
    }
    if (facts.current_catch_rate_kg_per_hour
            < profile.minimum_viable_catch_rate_kg_per_hour
        && state.relocation_count < profile.maximum_relocations) {
        return {false, false, true, "poor_catch_relocate"};
    }
    return {false, false, false, "continue_fishing"};
}

}  // namespace maritime::fisheries::autonomy::detail
