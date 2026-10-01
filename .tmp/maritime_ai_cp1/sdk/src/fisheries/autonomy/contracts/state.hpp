#pragma once

#include "maritime/fisheries/autonomy/fishing_skipper.hpp"

#include <deque>
#include <map>
#include <optional>

namespace maritime::fisheries::autonomy::detail {

struct GroundState {
    FishingGroundProfile profile;
    FishingGroundBelief belief;
    std::deque<FishingGroundObservation> observations;
};

struct RuntimeState {
    SkipperProfile profile;
    std::map<std::string, GroundState, std::less<>> grounds;
    VoyagePhase phase{VoyagePhase::InPort};
    AfterHaul after_haul{AfterHaul::Continue};
    std::string current_ground;
    std::uint64_t seed{};
    std::uint64_t next_decision_sequence{1};
    std::uint64_t next_receipt_sequence{1};
    std::uint32_t relocation_count{};
    std::optional<FishingDecision> outstanding_decision;
    std::vector<SkipperReceipt> receipts;
};

struct GroundScore {
    std::string ground_key;
    DecisionWhy why;
};

struct PolicyResult {
    bool secure_gear{};
    bool return_to_port{};
    bool relocate{};
    std::string reason;
};

struct RestoredState {
    RuntimeState runtime;
    std::string mission_snapshot;
};

}  // namespace maritime::fisheries::autonomy::detail
