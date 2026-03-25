# ==============================================================================
# CONSTANTS — Annual Forage (AF) Joint Optimizer
# ==============================================================================

FC_GREEN = '#5E9732'
FC_SLATE = '#5B707F'
COMMODITY_CODE = '0332'
INSURANCE_PLAN_CODE = 13
EXPECTED_INDEX = 100.0
RATE_YEAR = 2026  # Update annually — the crop year whose premium rates are used

COVERAGE_LEVELS = [0.90, 0.85, 0.80, 0.75, 0.70]
CAT_COVERAGE_LEVEL = 0.65

PRODUCTIVITY_FACTORS = [round(x / 100, 2) for x in range(150, 59, -1)]

DUAL_PURPOSE_STATES = ['48', '20', '40', '08', '35']  # TX, KS, OK, CO, NM

SEASON_LABELS = {
    1: 'August', 2: 'September', 3: 'October', 4: 'November',
    5: 'December', 6: 'January', 7: 'February', 8: 'March',
    9: 'April', 10: 'May', 11: 'June', 12: 'July',
}

AF_INTERVAL_MATRIX = {
    1:  {700: 'Sep-Oct', 701: 'Oct-Nov', 702: 'Nov-Dec', 703: 'Dec-Jan', 704: 'Jan-Feb', 705: 'Feb-Mar', 706: 'Sep-Mar'},
    2:  {707: 'Oct-Nov', 708: 'Nov-Dec', 709: 'Dec-Jan', 710: 'Jan-Feb', 711: 'Feb-Mar', 712: 'Mar-Apr', 713: 'Oct-Apr'},
    3:  {714: 'Nov-Dec', 715: 'Dec-Jan', 716: 'Jan-Feb', 717: 'Feb-Mar', 718: 'Mar-Apr', 719: 'Apr-May', 720: 'Nov-May'},
    4:  {721: 'Dec-Jan', 722: 'Jan-Feb', 723: 'Feb-Mar', 724: 'Mar-Apr', 725: 'Apr-May', 726: 'May-Jun', 727: 'Dec-Jun'},
    5:  {728: 'Jan-Feb', 729: 'Feb-Mar', 730: 'Mar-Apr', 731: 'Apr-May', 732: 'May-Jun', 733: 'Jun-Jul', 734: 'Jan-Jul'},
    6:  {735: 'Feb-Mar', 736: 'Mar-Apr', 737: 'Apr-May', 738: 'May-Jun', 739: 'Jun-Jul', 740: 'Jul-Aug', 741: 'Feb-Aug'},
    7:  {742: 'Mar-Apr', 743: 'Apr-May', 744: 'May-Jun', 745: 'Jun-Jul', 746: 'Jul-Aug', 747: 'Aug-Sep', 748: 'Mar-Sep'},
    8:  {749: 'Apr-May', 750: 'May-Jun', 751: 'Jun-Jul', 752: 'Jul-Aug', 753: 'Aug-Sep', 754: 'Sep-Oct', 755: 'Apr-Oct'},
    9:  {756: 'May-Jun', 757: 'Jun-Jul', 758: 'Jul-Aug', 759: 'Aug-Sep', 760: 'Sep-Oct', 761: 'Oct-Nov', 762: 'May-Nov'},
    10: {763: 'Jun-Jul', 764: 'Jul-Aug', 765: 'Aug-Sep', 766: 'Sep-Oct', 767: 'Oct-Nov', 768: 'Nov-Dec', 769: 'Jun-Dec'},
    11: {770: 'Jul-Aug', 771: 'Aug-Sep', 772: 'Sep-Oct', 773: 'Oct-Nov', 774: 'Nov-Dec', 775: 'Dec-Jan', 776: 'Jul-Jan'},
    12: {777: 'Aug-Sep', 778: 'Sep-Oct', 779: 'Oct-Nov', 780: 'Nov-Dec', 781: 'Dec-Jan', 782: 'Jan-Feb', 783: 'Aug-Feb'},
}

OPTIMIZER_CONSTRAINTS = {
    **{gs: {'max_weight': 0.45, 'min_weight': 0.10, 'num_intervals': 3, 'allow_fewer': False}
       for gs in range(1, 10)},
    **{gs: {'max_weight': 0.50, 'min_weight': 0.10, 'num_intervals': 3, 'allow_fewer': True, 'min_intervals': 2}
       for gs in range(10, 13)},
}

MONTH_NAMES_SHORT = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
MONTH_IDX = {m: i for i, m in enumerate(MONTH_NAMES_SHORT)}

# ==============================================================================
# HELPER FUNCTIONS (pure — no external dependencies)
# ==============================================================================


def get_buyup_intervals(growing_season):
    intervals = AF_INTERVAL_MATRIX.get(growing_season, {})
    codes = sorted(intervals.keys())
    return {c: intervals[c] for c in codes[:6]}


def get_cat_interval(growing_season):
    intervals = AF_INTERVAL_MATRIX.get(growing_season, {})
    codes = sorted(intervals.keys())
    if codes:
        cat_code = codes[-1]
        return {cat_code: intervals[cat_code]}
    return {}


def compute_shared_intervals(gs1, gs2):
    """Return the set of buy-up interval names shared between two growing seasons."""
    i1 = set(get_buyup_intervals(gs1).values())
    i2 = set(get_buyup_intervals(gs2).values())
    return i1 & i2


def compute_next_eligible_season(gs):
    """Given a growing season, find the next eligible season for same-dirt replanting.
    Walks forward through GS numbers until zero buy-up intervals overlap.
    Always 6 seasons forward due to the 6-of-12 interval structure."""
    for offset in range(1, 13):
        candidate = ((gs - 1 + offset) % 12) + 1
        if not compute_shared_intervals(gs, candidate):
            return candidate
    return None


def interval_to_months(interval_name):
    """Parse 'Sep-Oct' into (8, 9) — 0-indexed month numbers."""
    parts = interval_name.split('-')
    if len(parts) == 2:
        m1 = MONTH_IDX.get(parts[0])
        m2 = MONTH_IDX.get(parts[1])
        if m1 is not None and m2 is not None:
            return (m1, m2)
    return None
