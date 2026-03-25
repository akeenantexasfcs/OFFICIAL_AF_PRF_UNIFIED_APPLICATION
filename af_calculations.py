# ==============================================================================
# CALCULATIONS — Annual Forage (AF) USDA Rounding & Indemnity
# ==============================================================================

from decimal import Decimal, ROUND_HALF_UP
from af_constants import EXPECTED_INDEX


def _round_half_up(value, decimals=0):
    """
    Round using USDA convention (half rounds up), not Python's banker's rounding.
    Python's round(2.5) = 2, but USDA expects 3.
    This eliminates the $1 discrepancies against the Decision Support Tool.
    """
    d = Decimal(str(value))
    if decimals == 0:
        return float(d.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    else:
        quantize_to = Decimal('0.' + '0' * (decimals - 1) + '1')
        return float(d.quantize(quantize_to, rounding=ROUND_HALF_UP))


def dollar_amount_of_protection(cbv, coverage_level, productivity_factor):
    return cbv * coverage_level * productivity_factor


def policy_protection_per_interval(dollar_amount, insurable_interest, acres, interval_weight):
    return dollar_amount * insurable_interest * acres * interval_weight


def total_premium(policy_protection, premium_rate_per_100):
    return policy_protection * (premium_rate_per_100 / 100.0)


def premium_subsidy(total_prem, subsidy_percent):
    return total_prem * subsidy_percent


def producer_premium(total_prem, subsidy_amount):
    return total_prem - subsidy_amount


def indemnity(final_index, coverage_level, policy_protection):
    """Calculate indemnity using Decimal arithmetic to match USDA precision.
    Avoids floating-point division errors that cause $1 rounding discrepancies.
    Returns a float, but all intermediate math is done in Decimal.
    """
    trigger = EXPECTED_INDEX * coverage_level
    if final_index is not None and final_index < trigger:
        trigger_d = Decimal(str(trigger))
        index_d = Decimal(str(final_index))
        pp_d = Decimal(str(policy_protection))
        shortfall = Decimal('1') - index_d / trigger_d
        result = shortfall * pp_d
        # Round to nearest dollar using USDA half-up convention WHILE STILL IN DECIMAL
        return float(result.quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    return 0.0


def compute_interval_row(cbv, coverage_level, productivity_factor,
                         insurable_interest, acres, interval_weight,
                         base_rate, subsidy_pct, actual_index=None):
    # ================================================================
    # USDA/AgForce Rounding Convention (verified against GS-10 Grid 25318):
    #
    #   1. DA_full  = CBV * coverage * productivity  (full precision)
    #   2. DA_display = round_half_up(DA_full, 2)    (for ALL subsequent calcs)
    #   3. PP_raw   = DA_display * interest * acres * weight  (full precision)
    #   4. PP_display = round_half_up(PP_raw, 0)     (for DISPLAY only)
    #   5. per_acre_prem = DA_display * weight * BASE_RATE  (full precision)
    #   6. per_acre_sub  = per_acre_prem * subsidy          (full precision)
    #   7. Total Premium = round_half_up(per_acre_prem * acres, 0)
    #   8. Total Subsidy = round_half_up(per_acre_sub * acres, 0)
    #   9. Producer = TP - Subsidy  (derived)
    #  10. Indemnity = round_half_up((1 - index/trigger) * PP_from_da_full, 0)
    #
    # Key insights:
    #   - da_full (unrounded) feeds the protection value used for indemnity
    #   - da_display (rounded to 2dp) feeds premium calculations and PP display
    #   - This matches the optimizer engine (backtest_candidates_vectorized)
    #   - Subsidy is computed from per-acre precision, NOT from round(TP) * subsidy
    #   - All rounding uses USDA half-up convention, NOT Python banker's rounding
    #   - Indemnity shortfall uses Decimal arithmetic to avoid IEEE 754 errors
    # ================================================================
    da_full = dollar_amount_of_protection(cbv, coverage_level, productivity_factor)
    da_display = _round_half_up(da_full, 2)

    # PP display uses da_display (rounded)
    pp = _round_half_up(da_display * insurable_interest * acres * interval_weight, 0)

    # PP for indemnity uses da_full (unrounded) — matches optimizer vector math
    pp_for_indemnity = da_full * insurable_interest * acres * interval_weight

    # Premium and subsidy (uses rounded DA, computed per-acre first)
    per_acre_prem = da_display * insurable_interest * interval_weight * base_rate
    per_acre_sub = per_acre_prem * subsidy_pct

    tp = _round_half_up(per_acre_prem * acres, 0)
    ps = _round_half_up(per_acre_sub * acres, 0)
    prod = tp - ps  # Derived — not independently rounded

    result = {
        'dollar_amount': da_display,
        'policy_protection': pp,
        'total_premium': tp,
        'premium_subsidy': ps,
        'producer_premium': prod,
        'per_acre_premium': per_acre_prem,   # full precision, for totals
        'per_acre_subsidy': per_acre_sub,     # full precision, for totals
    }

    if actual_index is not None:
        # Indemnity uses PP from da_full (unrounded) — matches optimizer engine
        result['indemnity'] = indemnity(actual_index, coverage_level, pp_for_indemnity)
    else:
        result['indemnity'] = None

    return result
