"""Regression tests for the CPI parsing bug fixed in macro_collector.py.

Root cause: _cpi_window() returned the *full unfiltered* article text whenever
_CPI_BODY_RE failed to match the "tieu dung thang X tang/giam" sentence pattern
used by newer NSO releases. Older releases (2007-2009) phrase the headline
differently, so the window fallback let _MOM_RE/_YOY_RE match the wrong topic
entirely - e.g. picking up the gold/USD price change instead of CPI.
"""
from collectors.macro_collector import (
    _cpi_window,
    _MOM_RE,
    _YOY_RE,
    _MOM_RE_ANCHORED,
    _YOY_RE_ANCHORED_OLD,
)

# A real (paraphrased) excerpt from the May 2008 NSO release that triggered the bug:
# the CPI sentence states MoM only ("tang 3,91%"); the YoY figure that follows is
# actually about GOLD price, not CPI, and USD price MoM/YoY follows after that.
_MAY_2008_TEXT = (
    "Giá tiêu dùng tháng 5/2008 so với tháng trước tăng 3,91%, tăng cao nhất trong 5 "
    "tháng đầu năm nay. Giá tiêu dùng bình quân 5 tháng đầu năm 2008 tăng 19,09% so với "
    "5 tháng đầu năm 2007. Giá vàng tháng 5/2008 so với tháng trước giảm 3,9%, so với "
    "cùng kỳ năm trước tăng 32,45%. Giá đô la Mỹ tháng 5/2008 tăng 1,02% so với tháng "
    "trước, tăng 0,45% so với cùng kỳ năm trước."
)

# Sept 2008: CPI YoY *is* stated, with "so với cùng kỳ năm trước" preceding the value
# (no comma) - a wording _YOY_RE_OLD's comma requirement doesn't cover either.
_SEP_2008_TEXT = (
    "Giá tiêu dùng tháng 9/2008 tăng 0,18% so với tháng trước, là mức tăng thấp nhất "
    "trong vòng 17 tháng gần đây. Giá tiêu dùng tháng 9/2008 so với tháng 12 năm 2007 "
    "tăng 21,87%; so với cùng kỳ năm trước tăng 27,9%. Giá tiêu dùng bình quân 9 tháng "
    "đầu năm 2008 tăng 23%."
)

# Dec 2009: no narrative CPI MoM/YoY sentence at all - only the annual average vs
# 2008, plus gold/USD price changes. _CPI_BODY_RE should not match here.
_DEC_2009_TEXT = (
    "Chỉ số giá tiêu dùng bình quân năm 2009 tăng 6,88% so với bình quân năm 2008. "
    "Chỉ số giá vàng tháng 12/2009 tăng 10,49% so với tháng trước; tăng 64,32% so với "
    "cùng kỳ năm 2008. Chỉ số giá đô la Mỹ tháng 12/2009 tăng 3,19% so với tháng trước; "
    "tăng 10,7% so với cùng kỳ năm 2008."
)


def test_cpi_window_isolates_body_when_pattern_matches():
    # Modern wording puts the verb right after the month token ("thang N tang"),
    # unlike 2007-2009 releases which insert "so voi thang truoc" before it.
    modern_text = (
        "Chỉ số giá tiêu dùng (CPI) tháng 4/2026 tăng 0,3% so với tháng trước và "
        "tăng 3,2% so với cùng kỳ năm trước. Giá vàng tháng 4/2026 tăng 1,5%."
    )
    window = _cpi_window(modern_text)
    assert window.startswith("tiêu dùng (CPI) tháng 4/2026 tăng")
    assert "Giá vàng" not in window


def test_cpi_window_returns_empty_when_no_cpi_sentence_found():
    """Both the May 2008 and Dec 2009 wording styles have no sentence matching
    "tieu dung [tháng N] tang/giam" directly (the verb is separated from the
    month token by "so voi thang truoc"), so the window must come back empty -
    not fall back to the full text, which is what let the gold/USD-price
    sentence leak through and get parsed as if it were CPI."""
    assert _cpi_window(_MAY_2008_TEXT) == ""
    assert _cpi_window(_DEC_2009_TEXT) == ""


def test_unanchored_yoy_regex_matches_wrong_topic_on_may_2008_text():
    """Documents the original bug: the unanchored _YOY_RE, searched on the raw
    (unwindowed) text, grabs the USD-price sentence's YoY figure (0.45%) instead
    of leaving CPI YoY unmatched."""
    m = _YOY_RE.search(_MAY_2008_TEXT)
    assert m is not None
    assert m.group(2) == "0,45"  # USD price YoY, NOT a CPI figure


def test_anchored_mom_regex_extracts_correct_cpi_value():
    m = _MOM_RE_ANCHORED.search(_MAY_2008_TEXT)
    assert m is not None
    assert m.group(1) == "tăng"
    assert m.group(2) == "3,91"


def test_anchored_mom_regex_does_not_match_gold_or_usd_sentences():
    """The anchor ("Giá tiêu dùng tháng") must not match price sentences for
    other commodities even though they share the same "tăng X% so với tháng
    trước" suffix pattern."""
    gold_usd_only_text = (
        "Giá vàng tháng 5/2008 so với tháng trước giảm 3,9%. "
        "Giá đô la Mỹ tháng 5/2008 tăng 1,02% so với tháng trước."
    )
    assert _MOM_RE_ANCHORED.search(gold_usd_only_text) is None


def test_anchored_yoy_regex_extracts_correct_cpi_value_sep_2008():
    m = _YOY_RE_ANCHORED_OLD.search(_SEP_2008_TEXT)
    assert m is not None
    assert m.group(1) == "tăng"
    assert m.group(2) == "27,9"


def test_anchored_yoy_regex_does_not_match_gold_or_usd_sentences():
    gold_usd_only_text = (
        "Giá vàng tháng 12/2009 tăng 10,49% so với tháng trước; tăng 64,32% so với "
        "cùng kỳ năm 2008."
    )
    assert _YOY_RE_ANCHORED_OLD.search(gold_usd_only_text) is None
