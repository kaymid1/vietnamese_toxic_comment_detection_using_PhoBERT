from comment_crawl import _clean_comment_text, _is_comment_like_text


def test_clean_comment_strips_reaction_reply_report_minute_suffix() -> None:
    raw = (
        "Th\u01b0\u01a1ng em qu\u00e1, h\u00f4m qua t\u01b0\u1edfng em tho\u00e1t r\u1ed3i "
        "Th\u00edch Th\u00edch Ng\u1ea1c nhi\u00ean Bu\u1ed3n 1 1 Tr\u1ea3 l\u1eddi B\u00e1o vi ph\u1ea1m 34' tr\u01b0\u1edbc"
    )
    expected = "Th\u01b0\u01a1ng em qu\u00e1, h\u00f4m qua t\u01b0\u1edfng em tho\u00e1t r\u1ed3i"
    assert _clean_comment_text(raw) == expected


def test_clean_comment_keeps_normal_content() -> None:
    raw = "T\u1ed5 ch\u1ee9c s\u1eed d\u1ee5ng TPCMT theo khung h\u00ecnh ph\u1ea1t l\u00e0 7-15 n\u0103m r\u1ed3i em \u01a1i!"
    assert _clean_comment_text(raw) == raw


def test_ui_time_only_text_is_not_comment_like() -> None:
    assert _is_comment_like_text("34' tr\u01b0\u1edbc") is False
