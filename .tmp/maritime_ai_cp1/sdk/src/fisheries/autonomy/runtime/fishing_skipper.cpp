#include "maritime/fisheries/autonomy/fishing_skipper.hpp"

#include "../beliefs/belief_store.hpp"
#include "../contracts/state.hpp"
#include "../ground_selection/selector.hpp"
#include "../persistence/snapshot.hpp"
#include "../voyage_policy/policy.hpp"

#include <algorithm>
#include <utility>

namespace maritime::fisheries::autonomy {
namespace {

mission::MissionProfile voyage_mission_profile() {
    return {
        "autonomous_fishing_voyage",
        {
            {"prepare_voyage", {}, {{"Helm", true}, {"CrewPool", false}}},
            {"fishing_operation", {"prepare_voyage"},
             {{"Helm", true}, {"FishingGear", true}, {"Winch", true}}},
            {"processing_watch", {"prepare_voyage"},
             {{"ProcessingLine", false}, {"Cooling", false}}},
            {"return_land_close",
             {"fishing_operation", "processing_watch"},
             {{"Helm", true}}},
        },
    };
}

void append_receipt(
    detail::RuntimeState& state,
    const SkipperLimits& limits,
    std::string event,
    const ActionKind action,
    std::string ground,
    std::string reason) {
    if (state.receipts.size() >= limits.max_receipts) {
        state.receipts.erase(state.receipts.begin());
    }
    state.receipts.push_back({
        state.next_receipt_sequence++, std::move(event), action,
        std::move(ground), std::move(reason)});
}

FishingDecision make_decision(
    detail::RuntimeState& state,
    const SkipperLimits& limits,
    const ActionKind action,
    std::string ground,
    DecisionWhy why) {
    FishingDecision decision{
        state.next_decision_sequence++, action, std::move(ground),
        std::move(why)};
    append_receipt(
        state, limits, "decision_issued", decision.action,
        decision.ground_key, decision.why.reason);
    state.outstanding_decision = decision;
    return decision;
}

DecisionWhy reason(std::string value) {
    DecisionWhy why;
    why.reason = std::move(value);
    return why;
}

bool same_decision(
    const FishingDecision& expected,
    const FishingDecision& actual) {
    return expected.sequence == actual.sequence
        && expected.action == actual.action
        && expected.ground_key == actual.ground_key;
}

}  // namespace

class FishingSkipper::Impl {
public:
    explicit Impl(SkipperLimits value)
        : limits(std::move(value)), mission() {}

    FishingDecision decide_select(const VoyageFacts& facts) {
        const auto selected = detail::select_ground(state, facts);
        if (!selected) {
            state.phase = VoyagePhase::ReturnToPort;
            return make_decision(
                state, limits, ActionKind::ReturnToPort, {},
                reason("no_admitted_ground_return"));
        }
        return make_decision(
            state, limits, ActionKind::SelectGround,
            selected->ground_key, selected->why);
    }

    FishingDecision decide_operation(const VoyageFacts& facts) {
        const auto policy = detail::evaluate_policy(state, facts);
        if (policy.secure_gear) {
            state.after_haul = AfterHaul::Abort;
            return make_decision(
                state, limits, ActionKind::SecureAndHaul,
                state.current_ground, reason(policy.reason));
        }
        if (policy.relocate) {
            const auto alternative = detail::select_ground(
                state, facts, state.current_ground);
            state.after_haul = alternative
                ? AfterHaul::Relocate : AfterHaul::Return;
            return make_decision(
                state, limits, ActionKind::HaulGear,
                state.current_ground,
                reason(alternative ? policy.reason
                                   : "no_alternative_ground_return"));
        }
        if (policy.return_to_port) {
            state.after_haul = AfterHaul::Return;
            return make_decision(
                state, limits, ActionKind::HaulGear,
                state.current_ground, reason(policy.reason));
        }
        return make_decision(
            state, limits, ActionKind::OperateGear,
            state.current_ground, reason(policy.reason));
    }

    SkipperLimits limits;
    detail::RuntimeState state;
    mission::MissionService mission;
};

FishingSkipper::FishingSkipper(SkipperLimits limits)
    : impl_(new Impl(std::move(limits))) {}
FishingSkipper::~FishingSkipper() { delete impl_; }
FishingSkipper::FishingSkipper(FishingSkipper&& other) noexcept
    : impl_(std::exchange(other.impl_, nullptr)) {}
FishingSkipper& FishingSkipper::operator=(FishingSkipper&& other) noexcept {
    if (this == &other) return *this;
    delete impl_;
    impl_ = std::exchange(other.impl_, nullptr);
    return *this;
}

StartResult FishingSkipper::start_voyage(
    SkipperProfile profile,
    std::vector<FishingGroundProfile> grounds,
    const std::uint64_t seed) {
    if (!detail::validate_profile(profile)) {
        return {false, "invalid_skipper_profile", 0U};
    }
    detail::RuntimeState prepared;
    prepared.profile = std::move(profile);
    prepared.seed = seed;
    prepared.phase = VoyagePhase::PrepareVoyage;
    if (!detail::initialize_grounds(prepared, std::move(grounds),
                                    impl_->limits)) {
        return {false, "invalid_ground_profiles", 0U};
    }
    mission::MissionService prepared_mission;
    const auto candidate = prepared_mission.build_candidate(
        voyage_mission_profile());
    if (!candidate.accepted) {
        return {false, candidate.reason, 0U};
    }
    const auto commit = prepared_mission.commit_candidate(
        candidate.candidate_id);
    if (!commit.committed
        || prepared_mission.dispatch_ready()
            != std::vector<std::string>{"prepare_voyage"}) {
        return {false, "mission_prepare_dispatch_failed", 0U};
    }
    impl_->state = std::move(prepared);
    impl_->mission = std::move(prepared_mission);
    append_receipt(
        impl_->state, impl_->limits, "voyage_started",
        ActionKind::BeginVoyage, {}, "candidate_committed");
    return {true, {}, commit.mission_id};
}

bool FishingSkipper::admit_observation(
    const FishingGroundObservation& observation) {
    if (!detail::update_belief(
            impl_->state, observation, impl_->limits)) {
        return false;
    }
    append_receipt(
        impl_->state, impl_->limits, "ground_observation_admitted",
        ActionKind::Hold, observation.ground_key,
        "observation_bounded_belief_update");
    return true;
}

FishingDecision FishingSkipper::decide(const VoyageFacts& facts) {
    if (impl_->state.outstanding_decision) {
        return *impl_->state.outstanding_decision;
    }
    detail::age_beliefs(impl_->state, facts.tick);
    switch (impl_->state.phase) {
        case VoyagePhase::PrepareVoyage:
            return make_decision(impl_->state, impl_->limits,
                ActionKind::BeginVoyage, {}, reason("mission_ready"));
        case VoyagePhase::SelectGround:
            return impl_->decide_select(facts);
        case VoyagePhase::TransitToGround:
            return make_decision(impl_->state, impl_->limits,
                ActionKind::TransitToGround, impl_->state.current_ground,
                reason("transit_to_selected_ground"));
        case VoyagePhase::DeployGear:
            return make_decision(impl_->state, impl_->limits,
                ActionKind::DeployGear, impl_->state.current_ground,
                reason("ground_reached_deploy"));
        case VoyagePhase::OperateGear:
            return impl_->decide_operation(facts);
        case VoyagePhase::RecoverAndProcess:
            return make_decision(impl_->state, impl_->limits,
                ActionKind::RecoverAndProcess, impl_->state.current_ground,
                reason("recover_catch_and_queue_processing"));
        case VoyagePhase::ReturnToPort:
            return make_decision(impl_->state, impl_->limits,
                ActionKind::ReturnToPort, {}, reason("return_policy_committed"));
        case VoyagePhase::LandAndClose:
            return make_decision(impl_->state, impl_->limits,
                ActionKind::LandAndClose, {}, reason("port_reached_land_close"));
        case VoyagePhase::InPort:
        case VoyagePhase::Closed:
            return make_decision(impl_->state, impl_->limits,
                ActionKind::Hold, {}, reason("voyage_not_active"));
    }
    return make_decision(impl_->state, impl_->limits,
        ActionKind::Hold, {}, reason("invalid_phase"));
}

bool FishingSkipper::acknowledge(
    const FishingDecision& decision,
    const bool succeeded) {
    if (!impl_->state.outstanding_decision
        || !same_decision(*impl_->state.outstanding_decision, decision)) {
        return false;
    }
    append_receipt(
        impl_->state, impl_->limits,
        succeeded ? "action_completed" : "action_failed",
        decision.action, decision.ground_key, decision.why.reason);
    impl_->state.outstanding_decision.reset();
    if (!succeeded) return true;
    switch (decision.action) {
        case ActionKind::BeginVoyage:
            if (!impl_->mission.complete_task("prepare_voyage")) return false;
            impl_->mission.dispatch_ready();
            impl_->state.phase = VoyagePhase::SelectGround;
            break;
        case ActionKind::SelectGround:
            if (!impl_->state.grounds.contains(decision.ground_key)) return false;
            impl_->state.current_ground = decision.ground_key;
            impl_->state.phase = VoyagePhase::TransitToGround;
            break;
        case ActionKind::TransitToGround:
            impl_->state.phase = VoyagePhase::DeployGear;
            break;
        case ActionKind::DeployGear:
            impl_->state.phase = VoyagePhase::OperateGear;
            break;
        case ActionKind::OperateGear:
            break;
        case ActionKind::HaulGear:
        case ActionKind::SecureAndHaul:
            impl_->state.phase = VoyagePhase::RecoverAndProcess;
            break;
        case ActionKind::RecoverAndProcess:
            if (impl_->state.after_haul == AfterHaul::Relocate) {
                ++impl_->state.relocation_count;
                impl_->state.current_ground.clear();
                impl_->state.phase = VoyagePhase::SelectGround;
            } else {
                if (!impl_->mission.complete_task("fishing_operation")
                    || !impl_->mission.complete_task("processing_watch")) {
                    return false;
                }
                impl_->mission.dispatch_ready();
                impl_->state.phase = VoyagePhase::ReturnToPort;
            }
            break;
        case ActionKind::ReturnToPort:
            impl_->state.phase = VoyagePhase::LandAndClose;
            break;
        case ActionKind::LandAndClose:
            if (!impl_->mission.complete_task("return_land_close")) return false;
            impl_->state.phase = VoyagePhase::Closed;
            break;
        case ActionKind::Hold:
            break;
    }
    return true;
}

std::string FishingSkipper::save_snapshot() const {
    return detail::encode_snapshot(
        impl_->state, impl_->mission.save_snapshot());
}

bool FishingSkipper::restore_snapshot(const std::string_view payload) {
    auto restored = detail::decode_snapshot(payload, impl_->limits);
    if (!restored) return false;
    mission::MissionService restored_mission;
    if (!restored_mission.restore_snapshot(restored->mission_snapshot)) {
        return false;
    }
    if (!restored->runtime.current_ground.empty()
        && !restored->runtime.grounds.contains(
            restored->runtime.current_ground)) {
        return false;
    }
    restored->runtime.outstanding_decision.reset();
    impl_->state = std::move(restored->runtime);
    impl_->mission = std::move(restored_mission);
    append_receipt(
        impl_->state, impl_->limits, "fishing_autonomy_restored",
        ActionKind::Hold, impl_->state.current_ground,
        "snapshot_committed");
    return true;
}

VoyagePhase FishingSkipper::phase() const { return impl_->state.phase; }
std::string FishingSkipper::current_ground() const {
    return impl_->state.current_ground;
}
std::vector<FishingGroundBelief> FishingSkipper::beliefs() const {
    return detail::belief_views(impl_->state);
}
std::vector<SkipperReceipt> FishingSkipper::receipts() const {
    return impl_->state.receipts;
}
std::vector<mission::TaskView> FishingSkipper::mission_tasks() const {
    return impl_->mission.tasks();
}
std::vector<mission::Receipt> FishingSkipper::mission_receipts() const {
    return impl_->mission.receipts();
}

const char* to_string(const VoyagePhase value) noexcept {
    switch (value) {
        case VoyagePhase::InPort: return "InPort";
        case VoyagePhase::PrepareVoyage: return "PrepareVoyage";
        case VoyagePhase::SelectGround: return "SelectGround";
        case VoyagePhase::TransitToGround: return "TransitToGround";
        case VoyagePhase::DeployGear: return "DeployGear";
        case VoyagePhase::OperateGear: return "OperateGear";
        case VoyagePhase::RecoverAndProcess: return "RecoverAndProcess";
        case VoyagePhase::ReturnToPort: return "ReturnToPort";
        case VoyagePhase::LandAndClose: return "LandAndClose";
        case VoyagePhase::Closed: return "Closed";
    }
    return "Unknown";
}

const char* to_string(const ActionKind value) noexcept {
    switch (value) {
        case ActionKind::Hold: return "Hold";
        case ActionKind::BeginVoyage: return "BeginVoyage";
        case ActionKind::SelectGround: return "SelectGround";
        case ActionKind::TransitToGround: return "TransitToGround";
        case ActionKind::DeployGear: return "DeployGear";
        case ActionKind::OperateGear: return "OperateGear";
        case ActionKind::HaulGear: return "HaulGear";
        case ActionKind::SecureAndHaul: return "SecureAndHaul";
        case ActionKind::RecoverAndProcess: return "RecoverAndProcess";
        case ActionKind::ReturnToPort: return "ReturnToPort";
        case ActionKind::LandAndClose: return "LandAndClose";
    }
    return "Unknown";
}

}  // namespace maritime::fisheries::autonomy
