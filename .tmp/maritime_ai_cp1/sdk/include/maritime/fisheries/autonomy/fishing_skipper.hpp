#pragma once

#include "maritime/core/types.hpp"
#include "maritime/mission/mission_service.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace maritime::fisheries::autonomy {

inline constexpr std::string_view kSnapshotModule =
    "maritime.fishing_autonomy/1";

enum class VoyagePhase : std::uint8_t {
    InPort,
    PrepareVoyage,
    SelectGround,
    TransitToGround,
    DeployGear,
    OperateGear,
    RecoverAndProcess,
    ReturnToPort,
    LandAndClose,
    Closed,
};

enum class ActionKind : std::uint8_t {
    Hold,
    BeginVoyage,
    SelectGround,
    TransitToGround,
    DeployGear,
    OperateGear,
    HaulGear,
    SecureAndHaul,
    RecoverAndProcess,
    ReturnToPort,
    LandAndClose,
};

enum class AfterHaul : std::uint8_t {
    Continue,
    Relocate,
    Return,
    Abort,
};

struct SkipperLimits {
    std::size_t max_grounds{256};
    std::size_t max_observations_per_ground{32};
    std::size_t max_receipts{4096};
};

struct FishingGroundProfile {
    std::string key;
    double north_m{};
    double east_m{};
    double travel_cost_credits{};
    double prior_catch_rate_kg_per_hour{};
    double prior_bycatch_ratio{};
    double prior_market_value_per_kg{};
    bool access_allowed{true};
};

struct FishingGroundObservation {
    std::string ground_key;
    Tick tick{};
    double catch_rate_kg_per_hour{};
    double bycatch_ratio{};
    double market_value_per_kg{};
    bool access_allowed{true};
    bool closed{};
    double quality_risk_ratio{};
    double hold_risk_ratio{};
};

struct FishingGroundBelief {
    std::string ground_key;
    Tick last_observed_tick{};
    std::uint32_t observation_count{};
    double estimated_catch_rate_kg_per_hour{};
    double uncertainty_ratio{1.0};
    double estimated_bycatch_ratio{};
    double estimated_market_value_per_kg{};
    bool access_allowed{true};
    bool closed{};
    double quality_risk_ratio{};
    double hold_risk_ratio{};
    std::uint64_t revision{1};
};

struct SkipperProfile {
    std::string profile_id;
    double exploration_weight{0.35};
    double uncertainty_growth_per_tick{0.0005};
    double travel_cost_weight{1.0};
    double bycatch_risk_weight{25.0};
    double quality_risk_weight{15.0};
    double hold_risk_weight{20.0};
    double minimum_viable_catch_rate_kg_per_hour{20.0};
    double minimum_quota_remaining_kg{10.0};
    double maximum_bycatch_ratio{0.25};
    double maximum_hold_load_ratio{0.9};
    double minimum_quality_ratio{0.55};
    double maximum_processing_backlog_ratio{0.95};
    std::uint32_t maximum_relocations{3};
    std::uint32_t maximum_sets{3};
};

struct VoyageFacts {
    Tick tick{};
    double quota_remaining_kg{1000.0};
    double hold_load_ratio{};
    double weighted_quality_ratio{1.0};
    double processing_backlog_ratio{};
    double current_catch_rate_kg_per_hour{};
    double current_bycatch_ratio{};
    bool current_ground_accessible{true};
    bool current_ground_closed{};
    bool gear_faulted{};
    bool gear_deployed{};
    bool in_port{};
    std::uint32_t sets_completed{};
};

struct DecisionWhy {
    std::string reason;
    double expected_value{};
    double travel_penalty{};
    double bycatch_penalty{};
    double quality_penalty{};
    double hold_penalty{};
    double information_bonus{};
    double total_utility{};
};

struct FishingDecision {
    std::uint64_t sequence{};
    ActionKind action{ActionKind::Hold};
    std::string ground_key;
    DecisionWhy why;
};

struct SkipperReceipt {
    std::uint64_t sequence{};
    std::string event;
    ActionKind action{ActionKind::Hold};
    std::string ground_key;
    std::string reason;
};

struct StartResult {
    bool started{};
    std::string reason;
    std::uint64_t mission_id{};
};

class FishingSkipper {
public:
    explicit FishingSkipper(SkipperLimits limits = {});
    ~FishingSkipper();
    FishingSkipper(FishingSkipper&&) noexcept;
    FishingSkipper& operator=(FishingSkipper&&) noexcept;
    FishingSkipper(const FishingSkipper&) = delete;
    FishingSkipper& operator=(const FishingSkipper&) = delete;

    StartResult start_voyage(
        SkipperProfile profile,
        std::vector<FishingGroundProfile> grounds,
        std::uint64_t seed);
    bool admit_observation(const FishingGroundObservation& observation);
    FishingDecision decide(const VoyageFacts& facts);
    bool acknowledge(const FishingDecision& decision, bool succeeded);

    [[nodiscard]] std::string save_snapshot() const;
    bool restore_snapshot(std::string_view payload);

    [[nodiscard]] VoyagePhase phase() const;
    [[nodiscard]] std::string current_ground() const;
    [[nodiscard]] std::vector<FishingGroundBelief> beliefs() const;
    [[nodiscard]] std::vector<SkipperReceipt> receipts() const;
    [[nodiscard]] std::vector<mission::TaskView> mission_tasks() const;
    [[nodiscard]] std::vector<mission::Receipt> mission_receipts() const;

private:
    class Impl;
    Impl* impl_{};
};

[[nodiscard]] const char* to_string(VoyagePhase value) noexcept;
[[nodiscard]] const char* to_string(ActionKind value) noexcept;

}  // namespace maritime::fisheries::autonomy
