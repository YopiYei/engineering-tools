"""
Local-buckling classification of steel sections, AISC 360-16:

* Flexure     - Table B4.1b (compact / noncompact / slender)
* Compression - Table B4.1a (nonslender / slender)

Units: ksi and inches.

Every builder returns a list of element dictionaries; ``classify_section``
turns that list into the overall classification (the worst element governs).
"""

import math

COMPACT = "Compact"
NONCOMPACT = "Noncompact"
SLENDER = "Slender"
NONSLENDER = "Nonslender"

_RANK = {COMPACT: 0, NONSLENDER: 0, NONCOMPACT: 1, SLENDER: 2}


# ---------------------------------------------------------
# Core
# ---------------------------------------------------------

def _state(lam, lam_p, lam_r):
    if lam <= lam_p:
        return COMPACT
    if lam <= lam_r:
        return NONCOMPACT
    return SLENDER


def _element(
    name,
    case,
    lam,
    lam_tex,
    lam_p,
    lam_p_tex,
    lam_r,
    lam_r_tex,
):
    """
    Build one element record.

    ``*_tex`` strings are LaTeX fragments (already evaluated) used by the
    page to show the calculation.
    """

    return {
        "element": name,
        "case": case,
        "lam": lam,
        "lam_tex": lam_tex,
        "lam_p": lam_p,
        "lam_p_tex": lam_p_tex,
        "lam_r": lam_r,
        "lam_r_tex": lam_r_tex,
        "state": _state(lam, lam_p, lam_r),
    }


def _linear_limits(E, Fy, cp, cr):
    """
    Limits of the form  c * sqrt(E/Fy).
    """

    root = math.sqrt(E / Fy)
    return (
        cp * root,
        cr * root,
        rf"{cp}\sqrt{{E/F_y}} = {cp}\sqrt{{{E:g}/{Fy:g}}} = {cp * root:.2f}",
        rf"{cr}\sqrt{{E/F_y}} = {cr}\sqrt{{{E:g}/{Fy:g}}} = {cr * root:.2f}",
    )


def _check_inputs(E, Fy):
    if E <= 0 or Fy <= 0:
        raise ValueError("E and Fy must be greater than zero.")


# ---------------------------------------------------------
# I-shaped members (rolled W/M/S/HP or welded built-up)
# ---------------------------------------------------------

def i_shape_elements(
    bf,
    tf,
    h,
    tw,
    E,
    Fy,
    built_up=False,
    axis="strong",
    lam_f=None,
    lam_w=None,
):
    """
    Doubly symmetric I-shape.

    Strong axis: flange (case 10, or case 11 if built-up) + web (case 15).
    Weak axis:   flange only (case 13).

    ``lam_f`` / ``lam_w`` let the caller force the published AISC
    database values of bf/2tf and h/tw.
    """

    _check_inputs(E, Fy)

    if min(bf, tf, tw) <= 0 or h <= 0:
        raise ValueError("All dimensions must be greater than zero.")

    lam_f_value = lam_f if lam_f is not None else bf / (2 * tf)
    lam_f_tex = (
        rf"\lambda_f = \frac{{b_f}}{{2t_f}} = \frac{{{bf:g}}}{{2({tf:g})}} = {lam_f_value:.2f}"
        if lam_f is None
        else rf"\lambda_f = \frac{{b_f}}{{2t_f}} = {lam_f_value:.2f}\;\text{{(AISC database)}}"
    )

    elements = []

    if built_up and axis == "strong":

        lam_w_value = lam_w if lam_w is not None else h / tw
        kc = min(max(4 / math.sqrt(lam_w_value), 0.35), 0.76)
        FL = 0.7 * Fy
        lam_p, _, lam_p_tex, _ = _linear_limits(E, Fy, 0.38, 1.0)
        lam_r = 0.95 * math.sqrt(kc * E / FL)

        elements.append(
            _element(
                "Flange",
                "11",
                lam_f_value,
                lam_f_tex,
                lam_p,
                lam_p_tex,
                lam_r,
                rf"0.95\sqrt{{\frac{{k_c E}}{{F_L}}}} = 0.95\sqrt{{\frac{{({kc:.3f})({E:g})}}{{{FL:.1f}}}}} = {lam_r:.2f}"
                rf"\;\;\left(k_c = \frac{{4}}{{\sqrt{{h/t_w}}}},\; 0.35 \le k_c \le 0.76,\; F_L = 0.7F_y\right)",
            )
        )

    else:

        case = "10" if axis == "strong" else "13"
        lam_p, lam_r, lam_p_tex, lam_r_tex = _linear_limits(E, Fy, 0.38, 1.0)
        elements.append(
            _element("Flange", case, lam_f_value, lam_f_tex, lam_p, lam_p_tex, lam_r, lam_r_tex)
        )

    if axis == "strong":

        lam_w_value = lam_w if lam_w is not None else h / tw
        lam_p, lam_r, lam_p_tex, lam_r_tex = _linear_limits(E, Fy, 3.76, 5.70)
        lam_w_tex = (
            rf"\lambda_w = \frac{{h}}{{t_w}} = \frac{{{h:g}}}{{{tw:g}}} = {lam_w_value:.2f}"
            if lam_w is None
            else rf"\lambda_w = \frac{{h}}{{t_w}} = {lam_w_value:.2f}\;\text{{(AISC database)}}"
        )
        elements.append(
            _element("Web", "15", lam_w_value, lam_w_tex, lam_p, lam_p_tex, lam_r, lam_r_tex)
        )

    return elements


# ---------------------------------------------------------
# Channels
# ---------------------------------------------------------

def channel_elements(bf, tf, h, tw, E, Fy, axis="strong"):
    """
    Channel: flange b/tf with b = full flange width (case 10 / 13),
    web h/tw (case 15, strong axis only).
    """

    _check_inputs(E, Fy)

    if min(bf, tf, tw, h) <= 0:
        raise ValueError("All dimensions must be greater than zero.")

    lam_f = bf / tf
    lam_p, lam_r, lam_p_tex, lam_r_tex = _linear_limits(E, Fy, 0.38, 1.0)

    elements = [
        _element(
            "Flange",
            "10" if axis == "strong" else "13",
            lam_f,
            rf"\lambda_f = \frac{{b_f}}{{t_f}} = \frac{{{bf:g}}}{{{tf:g}}} = {lam_f:.2f}",
            lam_p,
            lam_p_tex,
            lam_r,
            lam_r_tex,
        )
    ]

    if axis == "strong":

        lam_w = h / tw
        lam_p, lam_r, lam_p_tex, lam_r_tex = _linear_limits(E, Fy, 3.76, 5.70)
        elements.append(
            _element(
                "Web",
                "15",
                lam_w,
                rf"\lambda_w = \frac{{h}}{{t_w}} = \frac{{{h:g}}}{{{tw:g}}} = {lam_w:.2f}",
                lam_p,
                lam_p_tex,
                lam_r,
                lam_r_tex,
            )
        )

    return elements


# ---------------------------------------------------------
# Tees
# ---------------------------------------------------------

def tee_elements(bf, tf, d, tw, E, Fy, lam_f=None, lam_s=None):
    """
    Tee: flange bf/2tf (case 10) and stem d/tw (case 14).
    """

    _check_inputs(E, Fy)

    if min(bf, tf, d, tw) <= 0:
        raise ValueError("All dimensions must be greater than zero.")

    lam_f_value = lam_f if lam_f is not None else bf / (2 * tf)
    lam_s_value = lam_s if lam_s is not None else d / tw

    p, r, p_tex, r_tex = _linear_limits(E, Fy, 0.38, 1.0)
    flange = _element(
        "Flange",
        "10",
        lam_f_value,
        rf"\lambda_f = \frac{{b_f}}{{2t_f}} = {lam_f_value:.2f}",
        p,
        p_tex,
        r,
        r_tex,
    )

    p, r, p_tex, r_tex = _linear_limits(E, Fy, 0.84, 1.52)
    stem = _element(
        "Stem",
        "14",
        lam_s_value,
        rf"\lambda_s = \frac{{d}}{{t_w}} = {lam_s_value:.2f}",
        p,
        p_tex,
        r,
        r_tex,
    )

    return [flange, stem]


# ---------------------------------------------------------
# Angles (single, or double with separators)
# ---------------------------------------------------------

def angle_elements(b, d, t, E, Fy):
    """
    Angle legs (case 12). b and d are the full leg lengths.
    """

    _check_inputs(E, Fy)

    if min(b, d, t) <= 0:
        raise ValueError("All dimensions must be greater than zero.")

    p, r, p_tex, r_tex = _linear_limits(E, Fy, 0.54, 0.91)

    elements = []

    for name, leg in (("Leg b", b), ("Leg d", d)):

        lam = leg / t
        symbol = "b" if name == "Leg b" else "d"
        elements.append(
            _element(
                name,
                "12",
                lam,
                rf"\lambda = \frac{{{symbol}}}{{t}} = \frac{{{leg:g}}}{{{t:g}}} = {lam:.2f}",
                p,
                p_tex,
                r,
                r_tex,
            )
        )

    return elements


# ---------------------------------------------------------
# Rectangular / square HSS
# ---------------------------------------------------------

def hss_rect_elements(b, h, t, E, Fy, axis="strong", lam_b=None, lam_h=None):
    """
    Rectangular HSS. ``b``: outside width (parallel to the bending axis),
    ``h``: outside depth, ``t``: design wall thickness.

    Flat widths are approximated as (dimension - 3t) unless the published
    b/t, h/t values are passed in.
    """

    _check_inputs(E, Fy)

    if min(b, h, t) <= 0:
        raise ValueError("All dimensions must be greater than zero.")

    if b - 3 * t <= 0 or h - 3 * t <= 0:
        raise ValueError("Wall thickness is too large for the outside dimensions.")

    lam_b_value = lam_b if lam_b is not None else (b - 3 * t) / t
    lam_h_value = lam_h if lam_h is not None else (h - 3 * t) / t

    # Compression flange is the face perpendicular to the bending plane.
    if axis == "strong":
        flange_lam, web_lam = lam_b_value, lam_h_value
        flange_sym, web_sym = "b", "h"
    else:
        flange_lam, web_lam = lam_h_value, lam_b_value
        flange_sym, web_sym = "h", "b"

    p, r, p_tex, r_tex = _linear_limits(E, Fy, 1.12, 1.40)
    flange = _element(
        "Flange",
        "17",
        flange_lam,
        rf"\lambda_f = \frac{{{flange_sym}_{{flat}}}}{{t}} = {flange_lam:.2f}",
        p,
        p_tex,
        r,
        r_tex,
    )

    p, r, p_tex, r_tex = _linear_limits(E, Fy, 2.42, 5.70)
    web = _element(
        "Web",
        "18",
        web_lam,
        rf"\lambda_w = \frac{{{web_sym}_{{flat}}}}{{t}} = {web_lam:.2f}",
        p,
        p_tex,
        r,
        r_tex,
    )

    return [flange, web]


# ---------------------------------------------------------
# Round HSS / pipe
# ---------------------------------------------------------

def hss_round_elements(D, t, E, Fy, lam=None):
    """
    Round HSS / pipe, D/t (case 19).
    """

    _check_inputs(E, Fy)

    if D <= 0 or t <= 0:
        raise ValueError("All dimensions must be greater than zero.")

    lam_value = lam if lam is not None else D / t

    ratio = E / Fy
    lam_p = 0.07 * ratio
    lam_r = 0.31 * ratio

    return [
        _element(
            "Wall",
            "19",
            lam_value,
            rf"\lambda = \frac{{D}}{{t}} = {lam_value:.2f}",
            lam_p,
            rf"0.07\frac{{E}}{{F_y}} = 0.07\frac{{{E:g}}}{{{Fy:g}}} = {lam_p:.2f}",
            lam_r,
            rf"0.31\frac{{E}}{{F_y}} = 0.31\frac{{{E:g}}}{{{Fy:g}}} = {lam_r:.2f}",
        )
    ]


# ---------------------------------------------------------
# AISC database dispatch
# ---------------------------------------------------------

def aisc_shape_elements(family_key, props, E, Fy, axis="strong"):
    """
    Build the element list for a shape taken from the AISC database
    (the published slenderness ratios are used when available).
    """

    if family_key == "w_m_s_hp":
        return i_shape_elements(
            props["bf"],
            props["tf"],
            props["h/tw"] * props["tw"],
            props["tw"],
            E,
            Fy,
            built_up=False,
            axis=axis,
            lam_f=props["bf/(2*tf)"],
            lam_w=props["h/tw"],
        )

    if family_key == "c_mc":
        h = props["d"] - 2 * props["k"]
        return channel_elements(
            props["bf"], props["tf"], h, props["tw"], E, Fy, axis=axis
        )

    if family_key == "wt_mt_st":
        return tee_elements(
            props["bf"],
            props["tf"],
            props["d"],
            props["tw"],
            E,
            Fy,
            lam_f=props["bf/(2*tf)"],
            lam_s=props["d/tw"],
        )

    if family_key in ("single_angles", "double_angles"):
        return angle_elements(props["b"], props["d"], props["t"], E, Fy)

    if family_key == "hss_rect":
        return hss_rect_elements(
            props["b"],
            props["h"],
            props["t(des)"],
            E,
            Fy,
            axis=axis,
            lam_b=props["b/t"],
            lam_h=props["h/t"],
        )

    if family_key == "hss_round":
        return hss_round_elements(
            props["OD"], props["t(des)"], E, Fy, lam=props["d/t"]
        )

    raise ValueError(f"Unknown shape family: {family_key}")


# ---------------------------------------------------------
# Overall classification
# ---------------------------------------------------------

def classify_section(elements):
    """
    The section is classified by its least favourable element.
    """

    if not elements:
        raise ValueError("No elements to classify.")

    governing = max(
        elements,
        key=lambda e: (_RANK[e["state"]], e["lam"] / (e["lam_p"] or e["lam_r"])),
    )

    return {
        "classification": governing["state"],
        "governing_element": governing["element"],
        "elements": elements,
    }


# =========================================================
# COMPRESSION - Table B4.1a (single limit, lambda_r)
# =========================================================

def _element_comp(name, case, lam, lam_tex, lam_r, lam_r_tex):
    """
    Element for axial compression: only slender / nonslender.
    """

    return {
        "element": name,
        "case": case,
        "lam": lam,
        "lam_tex": lam_tex,
        "lam_p": None,
        "lam_p_tex": None,
        "lam_r": lam_r,
        "lam_r_tex": lam_r_tex,
        "state": NONSLENDER if lam <= lam_r else SLENDER,
    }


def _comp_limit(E, Fy, c):
    root = math.sqrt(E / Fy)
    return (
        c * root,
        rf"{c}\sqrt{{E/F_y}} = {c}\sqrt{{{E:g}/{Fy:g}}} = {c * root:.2f}",
    )


def i_shape_compression(bf, tf, h, tw, E, Fy, built_up=False, lam_f=None, lam_w=None):
    """
    I-shape: flange (case 1, or case 2 if built-up) + web (case 5).
    """

    _check_inputs(E, Fy)

    if min(bf, tf, tw, h) <= 0:
        raise ValueError("All dimensions must be greater than zero.")

    lam_f_value = lam_f if lam_f is not None else bf / (2 * tf)
    lam_w_value = lam_w if lam_w is not None else h / tw

    lam_f_tex = (
        rf"\lambda_f = \frac{{b_f}}{{2t_f}} = \frac{{{bf:g}}}{{2({tf:g})}} = {lam_f_value:.2f}"
        if lam_f is None
        else rf"\lambda_f = \frac{{b_f}}{{2t_f}} = {lam_f_value:.2f}\;\text{{(AISC database)}}"
    )
    lam_w_tex = (
        rf"\lambda_w = \frac{{h}}{{t_w}} = \frac{{{h:g}}}{{{tw:g}}} = {lam_w_value:.2f}"
        if lam_w is None
        else rf"\lambda_w = \frac{{h}}{{t_w}} = {lam_w_value:.2f}\;\text{{(AISC database)}}"
    )

    if built_up:
        kc = min(max(4 / math.sqrt(lam_w_value), 0.35), 0.76)
        lam_r = 0.64 * math.sqrt(kc * E / Fy)
        flange = _element_comp(
            "Flange", "2", lam_f_value, lam_f_tex, lam_r,
            rf"0.64\sqrt{{\frac{{k_c E}}{{F_y}}}} = 0.64\sqrt{{\frac{{({kc:.3f})({E:g})}}{{{Fy:g}}}}} = {lam_r:.2f}"
            rf"\;\;\left(k_c = \frac{{4}}{{\sqrt{{h/t_w}}}},\; 0.35 \le k_c \le 0.76\right)",
        )
    else:
        lam_r, tex = _comp_limit(E, Fy, 0.56)
        flange = _element_comp("Flange", "1", lam_f_value, lam_f_tex, lam_r, tex)

    lam_r, tex = _comp_limit(E, Fy, 1.49)
    web = _element_comp("Web", "5", lam_w_value, lam_w_tex, lam_r, tex)

    return [flange, web]


def channel_compression(bf, tf, h, tw, E, Fy):
    """
    Channel: flange b/tf with b = full flange width (case 1), web (case 5).
    """

    _check_inputs(E, Fy)

    if min(bf, tf, tw, h) <= 0:
        raise ValueError("All dimensions must be greater than zero.")

    lam_f = bf / tf
    lam_w = h / tw

    lam_r, tex = _comp_limit(E, Fy, 0.56)
    flange = _element_comp(
        "Flange", "1", lam_f,
        rf"\lambda_f = \frac{{b_f}}{{t_f}} = \frac{{{bf:g}}}{{{tf:g}}} = {lam_f:.2f}",
        lam_r, tex,
    )

    lam_r, tex = _comp_limit(E, Fy, 1.49)
    web = _element_comp(
        "Web", "5", lam_w,
        rf"\lambda_w = \frac{{h}}{{t_w}} = \frac{{{h:g}}}{{{tw:g}}} = {lam_w:.2f}",
        lam_r, tex,
    )

    return [flange, web]


def tee_compression(bf, tf, d, tw, E, Fy, lam_f=None, lam_s=None):
    """
    Tee: flange (case 1) + stem (case 4).
    """

    _check_inputs(E, Fy)

    if min(bf, tf, d, tw) <= 0:
        raise ValueError("All dimensions must be greater than zero.")

    lam_f_value = lam_f if lam_f is not None else bf / (2 * tf)
    lam_s_value = lam_s if lam_s is not None else d / tw

    lam_r, tex = _comp_limit(E, Fy, 0.56)
    flange = _element_comp(
        "Flange", "1", lam_f_value,
        rf"\lambda_f = \frac{{b_f}}{{2t_f}} = {lam_f_value:.2f}", lam_r, tex,
    )

    lam_r, tex = _comp_limit(E, Fy, 0.75)
    stem = _element_comp(
        "Stem", "4", lam_s_value,
        rf"\lambda_s = \frac{{d}}{{t_w}} = {lam_s_value:.2f}", lam_r, tex,
    )

    return [flange, stem]


def angle_compression(b, d, t, E, Fy):
    """
    Angle legs, case 3 (also used for double angles; for double angles in
    continuous contact case 1 would apply, so case 3 is conservative there).
    """

    _check_inputs(E, Fy)

    if min(b, d, t) <= 0:
        raise ValueError("All dimensions must be greater than zero.")

    lam_r, tex = _comp_limit(E, Fy, 0.45)

    elements = []

    for name, symbol, leg in (("Leg b", "b", b), ("Leg d", "d", d)):
        lam = leg / t
        elements.append(
            _element_comp(
                name, "3", lam,
                rf"\lambda = \frac{{{symbol}}}{{t}} = \frac{{{leg:g}}}{{{t:g}}} = {lam:.2f}",
                lam_r, tex,
            )
        )

    return elements


def hss_rect_compression(b, h, t, E, Fy, lam_b=None, lam_h=None):
    """
    Rectangular HSS: all four walls, case 6.
    """

    _check_inputs(E, Fy)

    if min(b, h, t) <= 0:
        raise ValueError("All dimensions must be greater than zero.")

    if b - 3 * t <= 0 or h - 3 * t <= 0:
        raise ValueError("Wall thickness is too large for the outside dimensions.")

    lam_b_value = lam_b if lam_b is not None else (b - 3 * t) / t
    lam_h_value = lam_h if lam_h is not None else (h - 3 * t) / t

    lam_r, tex = _comp_limit(E, Fy, 1.40)

    return [
        _element_comp(
            "Wall b", "6", lam_b_value,
            rf"\lambda_b = \frac{{b_{{flat}}}}{{t}} = {lam_b_value:.2f}", lam_r, tex,
        ),
        _element_comp(
            "Wall h", "6", lam_h_value,
            rf"\lambda_h = \frac{{h_{{flat}}}}{{t}} = {lam_h_value:.2f}", lam_r, tex,
        ),
    ]


def hss_round_compression(D, t, E, Fy, lam=None):
    """
    Round HSS / pipe, D/t, case 9.
    """

    _check_inputs(E, Fy)

    if D <= 0 or t <= 0:
        raise ValueError("All dimensions must be greater than zero.")

    lam_value = lam if lam is not None else D / t
    lam_r = 0.11 * E / Fy

    return [
        _element_comp(
            "Wall", "9", lam_value,
            rf"\lambda = \frac{{D}}{{t}} = {lam_value:.2f}",
            lam_r,
            rf"0.11\frac{{E}}{{F_y}} = 0.11\frac{{{E:g}}}{{{Fy:g}}} = {lam_r:.2f}",
        )
    ]


def aisc_shape_compression(family_key, props, E, Fy):
    """
    Compression element list for a shape from the AISC database.
    """

    if family_key == "w_m_s_hp":
        return i_shape_compression(
            props["bf"], props["tf"], props["h/tw"] * props["tw"], props["tw"],
            E, Fy, built_up=False,
            lam_f=props["bf/(2*tf)"], lam_w=props["h/tw"],
        )

    if family_key == "c_mc":
        return channel_compression(
            props["bf"], props["tf"], props["d"] - 2 * props["k"], props["tw"], E, Fy
        )

    if family_key == "wt_mt_st":
        return tee_compression(
            props["bf"], props["tf"], props["d"], props["tw"], E, Fy,
            lam_f=props["bf/(2*tf)"], lam_s=props["d/tw"],
        )

    if family_key in ("single_angles", "double_angles"):
        return angle_compression(props["b"], props["d"], props["t"], E, Fy)

    if family_key == "hss_rect":
        return hss_rect_compression(
            props["b"], props["h"], props["t(des)"], E, Fy,
            lam_b=props["b/t"], lam_h=props["h/t"],
        )

    if family_key == "hss_round":
        return hss_round_compression(props["OD"], props["t(des)"], E, Fy, lam=props["d/t"])

    raise ValueError(f"Unknown shape family: {family_key}")
