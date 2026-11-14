"""NSO wording that changed in 2026 (texts copied from the real releases).

From May 2026 the CPI page is a two-line summary and the detail lives in the
attached PDF; the Q2/Q3 labour and socio-economic releases phrase their
headline sentences differently from Q1.
"""
from datetime import date

from collectors.macro_collector import (
    _INVESTMENT_RE,
    _INVESTMENT_RE_QUARTER,
    _LABOR_FORCE_RE,
    _LABOR_UNEMPLOYMENT_RE_ALT,
    _cpi_extras,
    _parse_period,
)

# October 2026 PDF (pdfplumber text): the footnote "3" is glued to the word.
_OCT_2026_PDF = (
    "III. LẠM PHÁT CƠ BẢN Lạm phát cơ bản3 tháng 10/2026 giảm 0,05% so với tháng trước và tăng 4,04% "
    "so với cùng kỳ năm trước. Bình quân 10 tháng năm 2026, lạm phát cơ bản tăng 4,24% "
    "Trong đó, chỉ số giá nhóm lương thực giảm 0,57%, góp phần giảm 0,03 điểm phần trăm; "
    "1.1. Lương thực (-0,57%) Chỉ số giá nhóm lương thực tháng 10/2026 giảm 0,57% so với tháng trước "
    "7. Giao thông (+2,66%) Chỉ số giá nhóm giao thông tháng 10/2026 tăng 2,66% so với tháng trước, "
    "- Nhóm giao thông tăng cao nhất với mức 15,57%, tác động làm CPI chung tăng 1,55 điểm phần trăm."
)

# June 2026 web page: the monthly figure comes first, a 6-month average later
# uses the "Chỉ số giá nhóm giao thông" wording the old code preferred.
_JUN_2026_HTML = (
    "Bốn nhóm hàng hóa và dịch vụ có chỉ số giá giảm gồm: (i) Nhóm giao thông giảm mạnh 4,85% so với "
    "tháng trước (góp phần làm CPI chung giảm 0,48 điểm phần trăm). ... Bình quân 6 tháng ... "
    "– Chỉ số giá nhóm giao thông tăng 5,23%, làm CPI chung tăng 0,52 điểm phần trăm"
)


def test_cpi_extras_read_the_2026_pdf_wording():
    assert _cpi_extras(_OCT_2026_PDF) == {
        "core_inflation_yoy": 4.04, "cpi_food": -0.57, "cpi_transport": 2.66,
    }


def test_transport_prefers_the_month_on_month_sentence():
    assert _cpi_extras(_JUN_2026_HTML)["cpi_transport"] == -4.85


def test_period_from_a_numeric_month_slug_with_quarter_words():
    url = ("https://www.nso.gov.vn/tin-tuc-thong-ke/2025/10/chi-so-gia-tieu-dung-chi-so-gia-vang-"
           "va-chi-so-gia-do-la-my-thang-9-quy-iii-va-9-thang-nam-2025/")
    assert _parse_period(url) == date(2025, 9, 1)


def test_labour_q3_2026_wording():
    text = ("Trong quý III/2026, tỷ lệ thất nghiệp trong độ tuổi lao động là 2,22%, tương đương quý trước. "
            "Lực lượng lao động từ 15 tuổi trở lên của cả nước quý III/2026 ước đạt 54,1 triệu người, cao hơn")
    assert _LABOR_UNEMPLOYMENT_RE_ALT.search(text).groups() == ("III", "2026", "2,22")
    assert _LABOR_FORCE_RE.search(text).groups() == ("III", "2026", "54,1")


def test_investment_quarter_in_brackets_of_a_9_month_release():
    text = ("Vốn đầu tư thực hiện toàn xã hội theo giá hiện hành chín tháng năm 2026, ước đạt 3.109,6 nghìn "
            "tỷ đồng, tăng 15,1% (quý III/2026 ước đạt 1.291,2 nghìn tỷ đồng, tăng 16,7% ) so với cùng kỳ")
    assert _INVESTMENT_RE.search(text) is None          # the cumulative figure is not taken
    assert _INVESTMENT_RE_QUARTER.search(text).groups() == ("III", "2026", "tăng", "16,7")


def test_monthly_report_trade_with_imports_derived():
    from collectors.macro_collector import parse_ktxh_trade
    text = ("Xuất khẩu hàng hóa tháng Mười đạt 64,67 tỷ USD, tăng 8,7% so với tháng trước ... "
            "cán cân thương mại hàng hóa tháng Mười xuất siêu 7,31 tỷ USD, sau khi xuất siêu 1,30 tỷ USD")
    got = {r["indicator"]: r["value"] for r in parse_ktxh_trade(text, date(2026, 10, 1), "u")}
    assert got == {"exports": 64.67, "imports": 57.36, "trade_balance": 7.31}
    deficit = text.replace("xuất siêu 7,31", "nhập siêu 2,5")
    assert {r["indicator"]: r["value"] for r in parse_ktxh_trade(deficit, date(2026, 10, 1), "u")}["imports"] == 67.17


def test_january_compares_with_december_not_the_gold_sentence():
    from collectors.macro_collector import _MOM_RE_JAN
    jan = ("chỉ số giá tiêu dùng (CPI) tháng 1/2020 tăng 1,23% so với tháng 12/2019 ... "
           "Chỉ số giá vàng tháng 1/2020 tăng 4,37% so với tháng trước")
    assert _MOM_RE_JAN.search(jan).groups() == ("tăng", "1,23")
    group = "nhóm giao thông giảm 2,12% so với tháng 12/2022"         # a group, not CPI
    assert _MOM_RE_JAN.search(group) is None


def test_cafef_bank_rate_table_and_summary():
    from collectors.macro_collector import parse_cafef_rates, summarize_rates
    head = "<tr><th>Ngân hàng</th><th>Không kỳ hạn</th><th>6 Tháng</th><th>12 Tháng</th></tr>"
    body = "".join(f"<tr><td>{b}</td><td>--</td><td>{r6} %</td><td>{r12} %</td></tr>"
                   for b, r6, r12 in (("A", "5", "6.9"), ("B", "3", "5.4"), ("C", "4", "--"),
                                      ("D", "5.5", "4.3"), ("E", "6", "6")))
    banks = parse_cafef_rates(f"<div><table class=x>{head}{body}</table></div>")
    assert banks[2] == {"bank": "C", "rates": {"6m": 4.0}}          # "--" = not offered
    got = {r["indicator"]: r["value"] for r in summarize_rates(banks, date(2026, 11, 14))}
    assert not any(k.startswith("deposit_12m") for k in got)          # only 4 banks quote 12m (< 5)
    assert (got["deposit_6m_max"], got["deposit_6m_min"], got["deposit_6m_avg"]) == (6.0, 3.0, 4.7)


def test_sjc_board():
    from collectors.macro_collector import parse_sjc
    payload = {"success": True, "latestDate": "08:31 14/11/2026", "data": [
        {"TypeName": "Vàng nhẫn SJC 99,99%", "BranchName": "Hồ Chí Minh", "BuyValue": 133.3e6, "SellValue": 137.3e6},
        {"TypeName": "Vàng SJC 1L, 10L, 1KG", "BranchName": "Hà Nội", "BuyValue": 1.0e8, "SellValue": 1.1e8},
        {"TypeName": "Vàng SJC 1L, 10L, 1KG", "BranchName": "Hồ Chí Minh", "BuyValue": 134.5e6, "SellValue": 137.5e6}]}
    got = {r["indicator"]: (r["value"], r["period"]) for r in parse_sjc(payload)}
    assert got == {"sjc_gold_buy": (134.5, date(2026, 11, 14)), "sjc_gold_sell": (137.5, date(2026, 11, 14))}
    assert parse_sjc({"success": False}) == []


def test_monthly_report_activity_figures():
    from collectors.macro_collector import parse_ktxh_activity
    text = ("Chỉ số sản xuất công nghiệp (IIP) tháng Mười ước tăng 2,3% so với tháng trước và tăng 15,0% so với "
            "cùng kỳ năm trước. Tính chung mười tháng năm 2026, chỉ số IIP tăng 12,4% ... Tổng mức bán lẻ hàng hóa "
            "và doanh thu dịch vụ tiêu dùng theo giá hiện hành tháng Mười ước đạt 695,5 nghìn tỷ đồng, giảm nhẹ 0,2% "
            "so với tháng trước và tăng 14,7% so với cùng kỳ năm trước. ... Trong tháng Tám, khách quốc tế đến nước "
            "ta ước đạt 1,99 triệu lượt người ... Trong tháng Mười, cả nước có gần 12,6 nghìn doanh nghiệp thành lập mới")
    got = {r["indicator"]: r["value"] for r in parse_ktxh_activity(text, date(2026, 10, 1), "u")}
    assert got == {"iip_yoy": 15.0, "retail_month": 695.5, "retail_yoy": 14.7,
                   "intl_visitors": 1.99, "new_firms": 12.6}
    plain = "Trong tháng Mười, cả nước có 4.726 doanh nghiệp thành lập mới"     # no "nghìn": a count
    assert parse_ktxh_activity(plain, date(2026, 10, 1), "u")[0]["value"] == 4.726


def test_report_month_from_slug_and_posting_date():
    from collectors.macro_collector import ktxh_period
    b = "https://www.nso.gov.vn/bai-top/"
    assert ktxh_period(b + "2026/10/bao-cao-tinh-hinh-kinh-te-xa-hoi-quy-iii-va-9-thang-nam-2026/") == date(2026, 9, 1)
    assert ktxh_period(b + "2026/04/bao-cao-tinh-hinh-kinh-te-xa-hoi-quy-i-nam-2026/") == date(2026, 3, 1)
    # posted 2026/06, slug wrongly says 2025: it is the May 2026 report
    assert ktxh_period(b + "2026/06/bao-cao-tinh-hinh-kinh-te-xa-hoi-thang-nam-va-5-thang-dau-nam-2025-2/") == date(2026, 5, 1)


def test_monthly_report_budget():
    from collectors.macro_collector import parse_ktxh_activity
    text = ("Tổng thu ngân sách nhà nước tháng Mười ước đạt 251,5 nghìn tỷ đồng. Lũy kế tổng thu ngân sách nhà nước "
            "mười tháng năm 2026 ước đạt 2.438,9 nghìn tỷ đồng ... Tổng chi ngân sách nhà nước tháng Mười ước đạt "
            "gần 235,0 nghìn tỷ đồng. Vốn đầu tư thực hiện từ nguồn ngân sách Nhà nước tháng 4/2024 ước đạt gần "
            "44 nghìn tỷ đồng")
    got = {r["indicator"]: r["value"] for r in parse_ktxh_activity(text, date(2026, 10, 1), "u")}
    assert got == {"budget_revenue": 251.5, "budget_spending": 235.0, "public_investment": 44.0}
