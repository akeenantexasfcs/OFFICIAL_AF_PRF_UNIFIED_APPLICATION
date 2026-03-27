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
    # Standard USDA RMA Rainfall Index Calculation Cascade
    #
    # Each step uses the ROUNDED output of the previous step.
    # This matches the official USDA Decision Support Tool exactly.
    #
    #   1. DA  = round(CBV * coverage * productivity, 2)
    #   2. PP  = round(DA * interest * acres * weight, 0)
    #   3. TP  = round(PP * base_rate, 0)
    #   4. PS  = round(TP * subsidy_pct, 0)
    #   5. Producer Premium = TP - PS  (derived, no rounding)
    #   6. Indemnity = round((1 - index/trigger) * PP, 0)
    # ================================================================

    # Step 1: Dollar Amount of Protection
    da_display = _round_half_up(cbv * coverage_level * productivity_factor, 2)

    # Step 2: Policy Protection (rounded, used for ALL downstream calcs)
    pp = _round_half_up(da_display * insurable_interest * acres * interval_weight, 0)

    # Step 3: Total Premium (from rounded PP)
    tp = _round_half_up(pp * base_rate, 0)

    # Step 4: Premium Subsidy (from rounded TP)
    ps = _round_half_up(tp * subsidy_pct, 0)

    # Step 5: Producer Premium (derived)
    prod = tp - ps

    result = {
        'dollar_amount': da_display,
        'policy_protection': pp,
        'total_premium': tp,
        'premium_subsidy': ps,
        'producer_premium': prod,
    }

    # Step 6: Indemnity (uses rounded PP)
    if actual_index is not None:
        result['indemnity'] = indemnity(actual_index, coverage_level, pp)
    else:
        result['indemnity'] = None

    return result
