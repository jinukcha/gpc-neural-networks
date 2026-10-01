#include "snapshot.hpp"

#include "../beliefs/belief_store.hpp"

#include <charconv>
#include <iomanip>
#include <sstream>

namespace maritime::fisheries::autonomy::detail {
namespace {

std::vector<std::string> split(
    const std::string_view value,
    const char delimiter) {
    std::vector<std::string> output;
    std::size_t start = 0U;
    while (start <= value.size()) {
        const auto end = value.find(delimiter, start);
        output.emplace_back(value.substr(start, end - start));
        if (end == std::string_view::npos) break;
        start = end + 1U;
    }
    return output;
}

template <typename Number>
bool parse_number(const std::string_view value, Number& output) {
    const auto result = std::from_chars(
        value.data(), value.data() + value.size(), output);
    return result.ec == std::errc{}
        && result.ptr == value.data() + value.size();
}

std::string hex_encode(const std::string_view value) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string output;
    output.reserve(value.size() * 2U);
    for (const unsigned char byte : value) {
        output.push_back(digits[byte >> 4U]);
        output.push_back(digits[byte & 0x0fU]);
    }
    return output;
}

std::optional<std::string> hex_decode(const std::string_view value) {
    if ((value.size() % 2U) != 0U) return std::nullopt;
    auto nibble = [](const char character) -> int {
        if (character >= '0' && character <= '9') return character - '0';
        if (character >= 'a' && character <= 'f') return character - 'a' + 10;
        if (character >= 'A' && character <= 'F') return character - 'A' + 10;
        return -1;
    };
    std::string output;
    output.reserve(value.size() / 2U);
    for (std::size_t index = 0U; index < value.size(); index += 2U) {
        const int high = nibble(value[index]);
        const int low = nibble(value[index + 1U]);
        if (high < 0 || low < 0) return std::nullopt;
        output.push_back(static_cast<char>((high << 4) | low));
    }
    return output;
}

bool parse_header(
    const std::vector<std::string>& fields,
    RuntimeState& state) {
    if (fields.size() != 9U) return false;
    unsigned phase{};
    unsigned after{};
    auto current = hex_decode(fields[7]);
    auto profile_id = hex_decode(fields[8]);
    if (!current || !profile_id
        || !parse_number(fields[1], state.seed)
        || !parse_number(fields[2], phase)
        || !parse_number(fields[3], after)
        || !parse_number(fields[4], state.relocation_count)
        || !parse_number(fields[5], state.next_decision_sequence)
        || !parse_number(fields[6], state.next_receipt_sequence)
        || phase > static_cast<unsigned>(VoyagePhase::Closed)
        || after > static_cast<unsigned>(AfterHaul::Abort)) {
        return false;
    }
    state.phase = static_cast<VoyagePhase>(phase);
    state.after_haul = static_cast<AfterHaul>(after);
    state.current_ground = std::move(*current);
    state.profile.profile_id = std::move(*profile_id);
    return true;
}

bool parse_profile(
    const std::vector<std::string>& fields,
    SkipperProfile& profile) {
    if (fields.size() != 15U) return false;
    return parse_number(fields[1], profile.exploration_weight)
        && parse_number(fields[2], profile.uncertainty_growth_per_tick)
        && parse_number(fields[3], profile.travel_cost_weight)
        && parse_number(fields[4], profile.bycatch_risk_weight)
        && parse_number(fields[5], profile.quality_risk_weight)
        && parse_number(fields[6], profile.hold_risk_weight)
        && parse_number(fields[7], profile.minimum_viable_catch_rate_kg_per_hour)
        && parse_number(fields[8], profile.minimum_quota_remaining_kg)
        && parse_number(fields[9], profile.maximum_bycatch_ratio)
        && parse_number(fields[10], profile.maximum_hold_load_ratio)
        && parse_number(fields[11], profile.minimum_quality_ratio)
        && parse_number(fields[12], profile.maximum_processing_backlog_ratio)
        && parse_number(fields[13], profile.maximum_relocations)
        && parse_number(fields[14], profile.maximum_sets);
}

bool parse_ground(
    const std::vector<std::string>& fields,
    RuntimeState& state) {
    if (fields.size() != 21U) return false;
    auto key = hex_decode(fields[1]);
    if (!key) return false;
    GroundState ground;
    ground.profile.key = *key;
    ground.belief.ground_key = *key;
    unsigned profile_access{};
    unsigned belief_access{};
    unsigned closed{};
    if (!parse_number(fields[2], ground.profile.north_m)
        || !parse_number(fields[3], ground.profile.east_m)
        || !parse_number(fields[4], ground.profile.travel_cost_credits)
        || !parse_number(fields[5], ground.profile.prior_catch_rate_kg_per_hour)
        || !parse_number(fields[6], ground.profile.prior_bycatch_ratio)
        || !parse_number(fields[7], ground.profile.prior_market_value_per_kg)
        || !parse_number(fields[8], profile_access)
        || !parse_number(fields[9], ground.belief.last_observed_tick)
        || !parse_number(fields[10], ground.belief.observation_count)
        || !parse_number(fields[11], ground.belief.estimated_catch_rate_kg_per_hour)
        || !parse_number(fields[12], ground.belief.uncertainty_ratio)
        || !parse_number(fields[13], ground.belief.estimated_bycatch_ratio)
        || !parse_number(fields[14], ground.belief.estimated_market_value_per_kg)
        || !parse_number(fields[15], belief_access)
        || !parse_number(fields[16], closed)
        || !parse_number(fields[17], ground.belief.quality_risk_ratio)
        || !parse_number(fields[18], ground.belief.hold_risk_ratio)
        || !parse_number(fields[19], ground.belief.revision)) {
        return false;
    }
    unsigned reserved{};
    if (!parse_number(fields[20], reserved) || reserved != 0U) return false;
    ground.profile.access_allowed = profile_access != 0U;
    ground.belief.access_allowed = belief_access != 0U;
    ground.belief.closed = closed != 0U;
    if (!validate_ground(ground.profile)
        || state.grounds.contains(ground.profile.key)) {
        return false;
    }
    state.grounds.emplace(ground.profile.key, std::move(ground));
    return true;
}

bool parse_receipt(
    const std::vector<std::string>& fields,
    RuntimeState& state) {
    if (fields.size() != 6U) return false;
    SkipperReceipt receipt;
    unsigned action{};
    auto event = hex_decode(fields[2]);
    auto ground = hex_decode(fields[4]);
    auto reason = hex_decode(fields[5]);
    if (!event || !ground || !reason
        || !parse_number(fields[1], receipt.sequence)
        || !parse_number(fields[3], action)
        || action > static_cast<unsigned>(ActionKind::LandAndClose)) {
        return false;
    }
    receipt.event = std::move(*event);
    receipt.action = static_cast<ActionKind>(action);
    receipt.ground_key = std::move(*ground);
    receipt.reason = std::move(*reason);
    state.receipts.push_back(std::move(receipt));
    return true;
}

}  // namespace

std::string encode_snapshot(
    const RuntimeState& state,
    const std::string_view mission_snapshot) {
    std::ostringstream output;
    output << std::setprecision(17);
    output << kSnapshotModule << '\n';
    output << "H|" << state.seed << '|'
           << static_cast<unsigned>(state.phase) << '|'
           << static_cast<unsigned>(state.after_haul) << '|'
           << state.relocation_count << '|'
           << state.next_decision_sequence << '|'
           << state.next_receipt_sequence << '|'
           << hex_encode(state.current_ground) << '|'
           << hex_encode(state.profile.profile_id) << '\n';
    const auto& profile = state.profile;
    output << "P|" << profile.exploration_weight << '|'
           << profile.uncertainty_growth_per_tick << '|'
           << profile.travel_cost_weight << '|'
           << profile.bycatch_risk_weight << '|'
           << profile.quality_risk_weight << '|'
           << profile.hold_risk_weight << '|'
           << profile.minimum_viable_catch_rate_kg_per_hour << '|'
           << profile.minimum_quota_remaining_kg << '|'
           << profile.maximum_bycatch_ratio << '|'
           << profile.maximum_hold_load_ratio << '|'
           << profile.minimum_quality_ratio << '|'
           << profile.maximum_processing_backlog_ratio << '|'
           << profile.maximum_relocations << '|'
           << profile.maximum_sets << '\n';
    for (const auto& entry : state.grounds) {
        const auto& ground = entry.second;
        const auto& belief = ground.belief;
        output << "G|" << hex_encode(ground.profile.key) << '|'
               << ground.profile.north_m << '|' << ground.profile.east_m << '|'
               << ground.profile.travel_cost_credits << '|'
               << ground.profile.prior_catch_rate_kg_per_hour << '|'
               << ground.profile.prior_bycatch_ratio << '|'
               << ground.profile.prior_market_value_per_kg << '|'
               << static_cast<unsigned>(ground.profile.access_allowed) << '|'
               << belief.last_observed_tick << '|' << belief.observation_count << '|'
               << belief.estimated_catch_rate_kg_per_hour << '|'
               << belief.uncertainty_ratio << '|'
               << belief.estimated_bycatch_ratio << '|'
               << belief.estimated_market_value_per_kg << '|'
               << static_cast<unsigned>(belief.access_allowed) << '|'
               << static_cast<unsigned>(belief.closed) << '|'
               << belief.quality_risk_ratio << '|'
               << belief.hold_risk_ratio << '|' << belief.revision << "|0\n";
    }
    for (const auto& receipt : state.receipts) {
        output << "R|" << receipt.sequence << '|'
               << hex_encode(receipt.event) << '|'
               << static_cast<unsigned>(receipt.action) << '|'
               << hex_encode(receipt.ground_key) << '|'
               << hex_encode(receipt.reason) << '\n';
    }
    output << "M|" << hex_encode(mission_snapshot) << '\n';
    return output.str();
}

std::optional<RestoredState> decode_snapshot(
    const std::string_view payload,
    const SkipperLimits& limits) {
    std::istringstream input{std::string(payload)};
    std::string line;
    if (!std::getline(input, line) || line != kSnapshotModule) {
        return std::nullopt;
    }
    RestoredState restored;
    bool saw_header = false;
    bool saw_profile = false;
    bool saw_mission = false;
    while (std::getline(input, line)) {
        const auto fields = split(line, '|');
        if (fields.empty()) continue;
        if (fields[0] == "H") {
            saw_header = parse_header(fields, restored.runtime);
            if (!saw_header) return std::nullopt;
        } else if (fields[0] == "P") {
            saw_profile = parse_profile(fields, restored.runtime.profile);
            if (!saw_profile) return std::nullopt;
        } else if (fields[0] == "G") {
            if (restored.runtime.grounds.size() >= limits.max_grounds
                || !parse_ground(fields, restored.runtime)) {
                return std::nullopt;
            }
        } else if (fields[0] == "R") {
            if (restored.runtime.receipts.size() >= limits.max_receipts
                || !parse_receipt(fields, restored.runtime)) {
                return std::nullopt;
            }
        } else if (fields[0] == "M" && fields.size() == 2U) {
            auto mission = hex_decode(fields[1]);
            if (!mission) return std::nullopt;
            restored.mission_snapshot = std::move(*mission);
            saw_mission = true;
        } else {
            return std::nullopt;
        }
    }
    if (!saw_header || !saw_profile || !saw_mission
        || restored.runtime.grounds.empty()
        || !validate_profile(restored.runtime.profile)) {
        return std::nullopt;
    }
    return restored;
}

}  // namespace maritime::fisheries::autonomy::detail
