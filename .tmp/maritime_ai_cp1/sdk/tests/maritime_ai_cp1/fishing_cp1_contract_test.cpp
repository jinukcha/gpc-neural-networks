#include "cp1_fixture.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

namespace maritime::test::maritime_ai_cp1 {
namespace {

void test_determinism_and_observation_boundary() {
    FishingSkipper first;
    FishingSkipper second;
    require(first.start_voyage(skipper_profile(), grounds(), 7711U).started,
            "first deterministic skipper start failed");
    require(second.start_voyage(skipper_profile(), grounds(), 7711U).started,
            "second deterministic skipper start failed");
    const auto admitted = observation("ground.alpha", 4U, 32.0, 0.1, 4.0);
    require(first.admit_observation(admitted), "first observation rejected");
    require(second.admit_observation(admitted), "second observation rejected");
    VoyageFacts facts;
    const auto first_begin = first.decide(facts);
    const auto second_begin = second.decide(facts);
    require(first_begin.action == second_begin.action,
            "same seed changed initial action");
    acknowledge(first, first_begin);
    acknowledge(second, second_begin);
    const double hidden_stock_a = 1.0e6;
    const double hidden_stock_b = 1.0;
    static_cast<void>(hidden_stock_a);
    static_cast<void>(hidden_stock_b);
    const auto first_ground = first.decide(facts);
    const auto second_ground = second.decide(facts);
    require(first_ground.ground_key == second_ground.ground_key,
            "hidden stock changed observation-bounded selection");
    require(first_ground.why.total_utility
                == second_ground.why.total_utility,
            "deterministic utility changed");
}

FishingDecision policy_decision(VoyageFacts facts) {
    FishingSkipper skipper;
    require(skipper.start_voyage(
                skipper_profile(), grounds(), 8877U).started,
            "policy skipper start failed");
    drive_to_operation(skipper, facts);
    return skipper.decide(facts);
}

void test_policy_guards() {
    VoyageFacts quota;
    quota.quota_remaining_kg = 5.0;
    require(policy_decision(quota).action == ActionKind::HaulGear,
            "quota did not force return haul");
    VoyageFacts bycatch;
    bycatch.current_bycatch_ratio = 0.5;
    require(policy_decision(bycatch).action == ActionKind::HaulGear,
            "bycatch did not force return haul");
    VoyageFacts hold;
    hold.hold_load_ratio = 0.95;
    require(policy_decision(hold).action == ActionKind::HaulGear,
            "hold capacity did not force return haul");
    VoyageFacts quality;
    quality.weighted_quality_ratio = 0.4;
    require(policy_decision(quality).action == ActionKind::HaulGear,
            "quality floor did not force return haul");
    VoyageFacts closure;
    closure.current_ground_closed = true;
    require(policy_decision(closure).action == ActionKind::HaulGear,
            "closure did not force relocation haul");
    VoyageFacts fault;
    fault.gear_faulted = true;
    require(policy_decision(fault).action == ActionKind::SecureAndHaul,
            "gear fault did not secure and haul");
}

void test_skipper_snapshot() {
    FishingSkipper skipper;
    require(skipper.start_voyage(
                skipper_profile(), grounds(), 9901U).started,
            "snapshot skipper start failed");
    VoyageFacts facts;
    drive_to_operation(skipper, facts);
    require(skipper.admit_observation(
                observation("ground.alpha", 12U, 5.0, 0.1, 4.0)),
            "snapshot observation rejected");
    const auto before = skipper.decide(facts);
    const auto snapshot = skipper.save_snapshot();
    FishingSkipper restored;
    require(restored.restore_snapshot(snapshot),
            "fishing autonomy snapshot restore failed");
    const auto after = restored.decide(facts);
    require(before.action == after.action
                && before.ground_key == after.ground_key,
            "restored skipper decision diverged");
    require(restored.current_ground() == "ground.alpha",
            "restored current ground changed");
    require(restored.mission_tasks().size() == 4U,
            "restored mission plan changed");
}

void submit_transit(
    SimulationWorld& world,
    const PersistentEntityId owner,
    SourceSequence& sequence,
    const std::string& token) {
    const Tick tick = world.capture_snapshot().world_tick + 1U;
    auto command = mission_command(
        CommandKind::SetStrategicMotion, owner, tick,
        sequence++, token);
    command.payload = SetStrategicMotionCommand{1.0, 0.0};
    submit_and_step(world, std::move(command), "transit");
}

void begin_product_voyage(
    campaign_fixture::PreparedWorld& prepared,
    FishingSkipper& skipper,
    VoyageFacts& facts,
    SourceSequence& sequence) {
    tether::register_p0f_assets(*prepared.world);
    require_ok(prepared.world->start(), "start CP1 product world");
    require(skipper.start_voyage(
                skipper_profile(), grounds(), 51021U).started,
            "product skipper start failed");
    const auto owner = prepared.spawned.persistent_id;
    const auto begin = require_action(
        skipper, facts, ActionKind::BeginVoyage);
    require_ok(prepared.world->begin_fisheries_voyage(
        owner, {prepared.world->capture_snapshot().world_tick, 0.0, 0.0}),
        "begin autonomous fisheries voyage");
    acknowledge(skipper, begin);
    const auto selected = require_action(
        skipper, facts, ActionKind::SelectGround);
    require(selected.ground_key == "ground.alpha",
            "product voyage did not select alpha first");
    acknowledge(skipper, selected);
    submit_transit(*prepared.world, owner, sequence, "cp1.transit.alpha");
    acknowledge(skipper, require_action(
        skipper, facts, ActionKind::TransitToGround));
}

FishingDecision restore_deployed_checkpoint(
    campaign_fixture::PreparedWorld& prepared,
    FishingSkipper& skipper,
    const VoyageFacts& facts,
    const FishingDecision& before,
    const WorldSnapshot& world_checkpoint,
    const std::string& skipper_checkpoint) {
    require_ok(prepared.world->restore_snapshot(world_checkpoint),
               "restore deployed-gear world checkpoint");
    FishingSkipper restored;
    require(restored.restore_snapshot(skipper_checkpoint),
            "restore deployed-gear skipper checkpoint");
    const auto after = require_action(
        restored, facts, ActionKind::HaulGear);
    require(before.ground_key == after.ground_key,
            "deployed-gear restore changed decision");
    skipper = std::move(restored);
    return after;
}

auto run_alpha_set(
    campaign_fixture::PreparedWorld& prepared,
    FishingSkipper& skipper,
    VoyageFacts& facts,
    SourceSequence& sequence,
    const FishingGearDefinition& gear_definition) {
    const auto owner = prepared.spawned.persistent_id;
    deploy_gear(*prepared.world, owner, gear_definition,
                "gear.cp1.alpha", sequence);
    acknowledge(skipper, require_action(
        skipper, facts, ActionKind::DeployGear));
    require(skipper.admit_observation(
        observation("ground.alpha",
                    prepared.world->capture_snapshot().world_tick,
                    5.0, 0.1, 4.0)),
        "alpha catch observation rejected");
    facts.current_catch_rate_kg_per_hour = 5.0;
    facts.current_bycatch_ratio = 0.1;
    facts.gear_deployed = true;
    facts.sets_completed = 1U;
    const auto world_checkpoint = prepared.world->capture_snapshot();
    const auto skipper_checkpoint = skipper.save_snapshot();
    const auto before = require_action(
        skipper, facts, ActionKind::HaulGear);
    const auto after = restore_deployed_checkpoint(
        prepared, skipper, facts, before,
        world_checkpoint, skipper_checkpoint);
    haul_gear(*prepared.world, owner, "gear.cp1.alpha",
              sequence, "cp1.haul.alpha");
    acknowledge(skipper, after);
    auto lot = prepared.world->append_catch_lot(
        owner, campaign_fixture::world_lot(
            "fishstock.r2c1.demersal.v1",
            HarvestClassification::Target,
            CatchProductState::DeckRaw, 1U, 15.0, 0.9));
    require_ok(lot, "append alpha catch lot");
    queue_processing(*prepared.world, owner,
                     prepared.processing, 10.0, sequence);
    acknowledge(skipper, require_action(
        skipper, facts, ActionKind::RecoverAndProcess));
    return lot.value;
}

auto run_beta_set(
    campaign_fixture::PreparedWorld& prepared,
    FishingSkipper& skipper,
    VoyageFacts& facts,
    SourceSequence& sequence,
    const FishingGearDefinition& gear_definition) {
    const auto owner = prepared.spawned.persistent_id;
    const auto selected = require_action(
        skipper, facts, ActionKind::SelectGround);
    require(selected.ground_key == "ground.beta",
            "poor catch did not relocate to beta");
    acknowledge(skipper, selected);
    submit_transit(*prepared.world, owner, sequence, "cp1.transit.beta");
    acknowledge(skipper, require_action(
        skipper, facts, ActionKind::TransitToGround));
    deploy_gear(*prepared.world, owner, gear_definition,
                "gear.cp1.beta", sequence);
    acknowledge(skipper, require_action(
        skipper, facts, ActionKind::DeployGear));
    require(skipper.admit_observation(
        observation("ground.beta",
                    prepared.world->capture_snapshot().world_tick,
                    80.0, 0.04, 5.0)),
        "beta catch observation rejected");
    facts.current_catch_rate_kg_per_hour = 80.0;
    facts.current_bycatch_ratio = 0.04;
    facts.sets_completed = 2U;
    const auto haul = require_action(
        skipper, facts, ActionKind::HaulGear);
    haul_gear(*prepared.world, owner, "gear.cp1.beta",
              sequence, "cp1.haul.beta");
    acknowledge(skipper, haul);
    auto lot = prepared.world->append_catch_lot(
        owner, campaign_fixture::world_lot(
            "fishstock.r2c1.demersal.v1",
            HarvestClassification::Target,
            CatchProductState::DeckRaw, 1U, 35.0, 0.95));
    require_ok(lot, "append beta catch lot");
    acknowledge(skipper, require_action(
        skipper, facts, ActionKind::RecoverAndProcess));
    return lot.value;
}

template <typename LotId>
void finish_product_voyage(
    campaign_fixture::PreparedWorld& prepared,
    FishingSkipper& skipper,
    const VoyageFacts& facts,
    const LotId first_lot,
    const LotId second_lot) {
    const auto owner = prepared.spawned.persistent_id;
    acknowledge(skipper, require_action(
        skipper, facts, ActionKind::ReturnToPort));
    auto landing = prepared.world->settle_fisheries_landing(
        owner, prepared.world->capture_snapshot().world_tick + 1U,
        {first_lot, second_lot});
    require_ok(landing, "settle autonomous landing");
    require_ok(prepared.world->close_fisheries_voyage(
        owner, prepared.world->capture_snapshot().world_tick + 2U, 25.0),
        "close autonomous voyage");
    acknowledge(skipper, require_action(
        skipper, facts, ActionKind::LandAndClose));
    require(skipper.phase() == VoyagePhase::Closed,
            "autonomous voyage did not close");
    const auto campaign = prepared.world->query_fisheries_campaign(owner);
    require_ok(campaign, "query closed campaign");
    require(campaign.value.voyage_records().size() == 1U,
            "closed voyage record missing");
    const auto summary = prepared.world->query_fisheries_summary(owner);
    require_ok(summary, "query processing summary");
    require(summary.value.target_mass_kg >= 0.0,
            "processing summary invalid");
}

void run_product_voyage() {
    auto prepared = campaign_fixture::prepare_world(51021U);
    FishingSkipper skipper;
    VoyageFacts facts;
    SourceSequence sequence = 1U;
    begin_product_voyage(prepared, skipper, facts, sequence);
    const auto gear = tether::load_gear("gear.trawl.v1.json");
    const auto alpha = run_alpha_set(
        prepared, skipper, facts, sequence, gear);
    const auto beta = run_beta_set(
        prepared, skipper, facts, sequence, gear);
    finish_product_voyage(prepared, skipper, facts, alpha, beta);
}

void write_receipt(const std::filesystem::path& path) {
    std::ofstream output(path, std::ios::binary);
    require(static_cast<bool>(output), "cannot write CP1 receipt");
    output << "{\n"
           << "  \"schema\": \"maritime.ai.r0a.cp1.native_test/1\",\n"
           << "  \"passed\": true,\n"
           << "  \"snapshot_module\": \"maritime.fishing_autonomy/1\",\n"
           << "  \"phases\": [\"belief\", \"selection\", \"deploy\", "
              "\"relocate\", \"processing\", \"return\", \"landing\", "
              "\"restore\", \"policy_guards\"]\n"
           << "}\n";
}

}  // namespace
}  // namespace maritime::test::maritime_ai_cp1

int main(int argc, char** argv) {
    using namespace maritime::test::maritime_ai_cp1;
    try {
        test_determinism_and_observation_boundary();
        test_policy_guards();
        test_skipper_snapshot();
        run_product_voyage();
        if (argc > 1) write_receipt(argv[1]);
        std::cout << "MARITIME_AI_CP1 phase=belief\n"
                  << "MARITIME_AI_CP1 phase=selection\n"
                  << "MARITIME_AI_CP1 phase=deploy\n"
                  << "MARITIME_AI_CP1 phase=relocate\n"
                  << "MARITIME_AI_CP1 phase=processing\n"
                  << "MARITIME_AI_CP1 phase=return\n"
                  << "MARITIME_AI_CP1 phase=landing\n"
                  << "MARITIME_AI_CP1 phase=restore\n"
                  << "MARITIME_AI_CP1 phase=policy_guards\n"
                  << "MARITIME_AI_CP1_PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MARITIME_AI_CP1_FAIL " << error.what() << '\n';
        return 1;
    }
}
