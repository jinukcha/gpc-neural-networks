#pragma once

#include "integration/campaign/fixtures.hpp"
#include "maritime/fisheries/autonomy/fishing_skipper.hpp"
#include "test_support.hpp"
#include "unit/tether/fixture.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace maritime::test::maritime_ai_cp1 {

using fisheries::autonomy::ActionKind;
using fisheries::autonomy::FishingDecision;
using fisheries::autonomy::FishingGroundObservation;
using fisheries::autonomy::FishingGroundProfile;
using fisheries::autonomy::FishingSkipper;
using fisheries::autonomy::SkipperProfile;
using fisheries::autonomy::VoyageFacts;
using fisheries::autonomy::VoyagePhase;

inline SkipperProfile skipper_profile() {
    SkipperProfile profile;
    profile.profile_id = "skipper.cp1.v1";
    profile.exploration_weight = 0.25;
    profile.minimum_viable_catch_rate_kg_per_hour = 20.0;
    profile.minimum_quota_remaining_kg = 10.0;
    profile.maximum_bycatch_ratio = 0.3;
    profile.maximum_hold_load_ratio = 0.9;
    profile.minimum_quality_ratio = 0.55;
    profile.maximum_processing_backlog_ratio = 0.95;
    profile.maximum_relocations = 2U;
    profile.maximum_sets = 2U;
    return profile;
}

inline std::vector<FishingGroundProfile> grounds() {
    return {
        {"ground.alpha", 0.0, 0.0, 1.0, 60.0, 0.08, 4.0, true},
        {"ground.beta", 250.0, 20.0, 5.0, 50.0, 0.05, 5.0, true},
        {"ground.closed", 500.0, 0.0, 2.0, 100.0, 0.01, 6.0, false},
    };
}

inline FishingGroundObservation observation(
    std::string key,
    const Tick tick,
    const double catch_rate,
    const double bycatch,
    const double value,
    const bool closed = false) {
    FishingGroundObservation result;
    result.ground_key = std::move(key);
    result.tick = tick;
    result.catch_rate_kg_per_hour = catch_rate;
    result.bycatch_ratio = bycatch;
    result.market_value_per_kg = value;
    result.closed = closed;
    result.access_allowed = !closed;
    result.quality_risk_ratio = 0.05;
    result.hold_risk_ratio = 0.05;
    return result;
}

inline FishingDecision require_action(
    FishingSkipper& skipper,
    const VoyageFacts& facts,
    const ActionKind expected) {
    const auto decision = skipper.decide(facts);
    require(decision.action == expected,
            std::string{"unexpected skipper action: "}
                + fisheries::autonomy::to_string(decision.action));
    return decision;
}

inline void acknowledge(
    FishingSkipper& skipper,
    const FishingDecision& decision) {
    require(skipper.acknowledge(decision, true),
            "skipper acknowledgement failed");
}

inline void drive_to_operation(
    FishingSkipper& skipper,
    VoyageFacts facts = {}) {
    acknowledge(skipper, require_action(
        skipper, facts, ActionKind::BeginVoyage));
    const auto selected = require_action(
        skipper, facts, ActionKind::SelectGround);
    require(selected.ground_key == "ground.alpha",
            "deterministic first ground changed");
    acknowledge(skipper, selected);
    acknowledge(skipper, require_action(
        skipper, facts, ActionKind::TransitToGround));
    acknowledge(skipper, require_action(
        skipper, facts, ActionKind::DeployGear));
    require(skipper.phase() == VoyagePhase::OperateGear,
            "skipper did not reach operation phase");
}

inline CommandEnvelope mission_command(
    const CommandKind kind,
    const PersistentEntityId owner,
    const Tick tick,
    const SourceSequence sequence,
    std::string token) {
    CommandEnvelope command;
    command.kind = kind;
    command.target = owner;
    command.target_tick = tick;
    command.phase_class = CommandPhase::Control;
    command.authority = AuthorityClass::MissionAi;
    command.source_id = 9101U;
    command.source_sequence = sequence;
    command.idempotency_token = std::move(token);
    return command;
}

inline void submit_and_step(
    SimulationWorld& world,
    CommandEnvelope command,
    const std::string_view label) {
    require(world.submit(std::move(command))
                == CommandIngressResult::AcceptedToIngress,
            std::string{label} + " command admission failed");
    require_ok(world.step_one_tick(),
               std::string{label} + " command step failed");
}

inline void deploy_gear(
    SimulationWorld& world,
    const PersistentEntityId owner,
    const FishingGearDefinition& definition,
    const std::string_view gear_key,
    SourceSequence& sequence) {
    const Tick tick = world.capture_snapshot().world_tick + 1U;
    auto command = mission_command(
        CommandKind::DeployFishingGear, owner, tick,
        sequence++, "cp1.deploy." + std::string{gear_key});
    command.payload = DeployFishingGearCommand{
        std::string{gear_key}, definition.key, {0.0, 0.0, 20.0}, 80.0};
    submit_and_step(world, std::move(command), "deploy gear");
    for (std::size_t index = 0U; index < 20U; ++index) {
        const auto state = world.query_fishing_gear(owner, gear_key);
        require_ok(state, "query deployed gear");
        if (state.value.operation_state
                == FishingGearOperationState::DeployedTowing
            || state.value.operation_state
                == FishingGearOperationState::DeployedSoaking) {
            return;
        }
        require_ok(world.step_one_tick(), "advance gear deployment");
    }
    throw Failure("gear did not deploy within bounded ticks");
}

inline void haul_gear(
    SimulationWorld& world,
    const PersistentEntityId owner,
    const std::string_view gear_key,
    SourceSequence& sequence,
    std::string token) {
    const Tick tick = world.capture_snapshot().world_tick + 1U;
    auto command = mission_command(
        CommandKind::HaulFishingGear, owner, tick,
        sequence++, std::move(token));
    command.payload = HaulFishingGearCommand{std::string{gear_key}};
    submit_and_step(world, std::move(command), "haul gear");
    for (std::size_t index = 0U; index < 20U; ++index) {
        const auto state = world.query_fishing_gear(owner, gear_key);
        require_ok(state, "query hauled gear");
        if (state.value.operation_state
            == FishingGearOperationState::Recovered) {
            return;
        }
        require_ok(world.step_one_tick(), "advance gear haul");
    }
    throw Failure("gear did not recover within bounded ticks");
}

inline void queue_processing(
    SimulationWorld& world,
    const PersistentEntityId owner,
    const ProcessingProfile& profile,
    const double mass_kg,
    SourceSequence& sequence) {
    const Tick tick = world.capture_snapshot().world_tick + 1U;
    auto availability = mission_command(
        CommandKind::SetCatchProcessingAvailability,
        owner, tick, sequence++, "cp1.processing.availability");
    availability.phase_class = CommandPhase::Loading;
    availability.payload = SetCatchProcessingAvailabilityCommand{1.0, 1.0};
    require(world.submit(std::move(availability))
                == CommandIngressResult::AcceptedToIngress,
            "processing availability admission failed");
    auto queue = mission_command(
        CommandKind::QueueCatchProcessing,
        owner, tick, sequence++, "cp1.processing.queue");
    queue.phase_class = CommandPhase::Loading;
    queue.payload = QueueCatchProcessingCommand{
        profile.key, profile.recipes.front().key, mass_kg};
    submit_and_step(world, std::move(queue), "queue processing");
}

}  // namespace maritime::test::maritime_ai_cp1
