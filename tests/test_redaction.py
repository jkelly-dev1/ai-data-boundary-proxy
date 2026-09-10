"""Redaction, and the two properties it can silently fail to have."""
from __future__ import annotations

import pytest

from boundary.corpus import ROSTER, build_corpus
from boundary.detectors import default_suite, detect_strict
from boundary.models import DataClass, RedactionStrategy
from boundary.redaction import (
    Pseudonymizer,
    Redactor,
    _partial,
    quasi_identifier_set,
    residual_classes,
)

KEY = b"test-key"


def test_spans_are_applied_right_to_left_so_offsets_stay_valid():
    """Forward application corrupts every span after the first substitution.

    This demonstrates WHY the descending order is necessary: the same findings
    applied left to right are computed here and shown to produce different,
    corrupted output, with the leading digits of the card left exposed.

    It does NOT exercise `Redactor.apply`'s own ordering, because the forward
    pass below is computed locally rather than by the redactor. The test that
    holds the redactor to this property is
    `test_no_detected_value_survives_redaction_in_any_corpus_document`, which
    asserts on the redactor's own output over the shipped corpus.
    """
    text = "SSN 482-19-7734 and card 4111 1111 1111 1111 on file."
    findings = detect_strict(text)
    assert len(findings) >= 2

    good = Redactor().apply(text, findings).text
    assert "482-19-7734" not in good
    assert "4111 1111 1111 1111" not in good

    # Deliberate wrong ordering, to prove the ordering is what protects it.
    out = text
    for f in sorted(findings, key=lambda f: f.span.start):
        out = out[: f.span.start] + "[X]" + out[f.span.end:]
    assert out != good
    # The second substitution lands at a stale offset, so it cuts the card
    # number in half and leaves the leading digits exposed.
    assert "4111" in out, f"forward application should leave debris: {out!r}"


def test_a_stable_pseudonym_lets_the_recipient_link_two_requests():
    """The disclosure pseudonymization does not remove.

    Mutation check, executed in-test: a different salt breaks the linkage,
    showing the linkage is a property of the keying and not of the format.
    """
    ps = Pseudonymizer(KEY)
    a = ps.token(DataClass.EMAIL, "jrivera@acme.example")
    b = ps.token(DataClass.EMAIL, " JRivera@Acme.Example ")
    assert a == b
    assert ps.links("jrivera@acme.example", "JRIVERA@ACME.EXAMPLE", DataClass.EMAIL)

    rotated = Pseudonymizer(KEY, salt="2026-08")
    assert rotated.token(DataClass.EMAIL, "jrivera@acme.example") != a


def test_pseudonym_tokens_do_not_reveal_the_value():
    ps = Pseudonymizer(KEY)
    token = ps.token(DataClass.SSN, "482-19-7734")
    assert "482" not in token
    assert "7734" not in token


def test_partial_redaction_of_a_card_keeps_only_the_last_four():
    """The exact output shape, so "keeps four" cannot quietly keep twelve.

    Mutation check, executed against `_partial`: keep 12 digits and the card
    line reads `card ****111111111111 ok`, which the exact comparison
    catches; keep 4 for every class and the phone line changes.
    """
    text = "card 4111 1111 1111 1111 ok"
    assert Redactor().apply(text, detect_strict(text)).text == "card ************1111 ok"
    assert _partial(DataClass.CREDIT_CARD, "4111 1111 1111 1111") == "************1111"
    assert _partial(DataClass.PHONE, "512-555-0143") == "********43"
    assert _partial(DataClass.POSTAL_CODE, "78") == "[REDACTED:postal_code]"


def test_pseudonym_without_a_key_falls_back_to_a_mask():
    """No key means no token, and never the raw value.

    Mutation check, executed against `_replacement`: return the value when
    the pseudonymizer is absent and the address comes through in cleartext.
    """
    text = "Contact jrivera@acme.example now."
    out = Redactor(pseudonymizer=None).apply(text, detect_strict(text))
    assert out.text == "Contact [REDACTED:email] now."
    assert out.count == 1


def test_drop_removes_the_span_entirely():
    """DROP leaves nothing where the value was, not a marker and not the value.

    Mutation check, executed against `_replacement`: return the value for DROP
    and the text is unchanged with a count of zero.
    """
    text = "SSN 482-19-7734 on file."
    out = Redactor(override=RedactionStrategy.DROP).apply(text, detect_strict(text))
    assert out.text == "SSN  on file."
    assert out.count == 1


def test_quasi_identifiers_survive_redaction_by_default():
    """Every direct identifier can go and the record still identifies someone.

    Mutation check, executed in-test: an override that masks everything is
    shown to remove them, proving the survival is the default strategy table
    rather than a limitation of the redactor.
    """
    suite = default_suite()
    text = "DOB: 1982-03-14. Sex: female. Zip: 78704. Title: Staff Engineer."
    findings = suite.run(text)
    redactor = Redactor(pseudonymizer=Pseudonymizer(KEY))
    survived = residual_classes(text, findings, redactor)
    assert survived, "quasi-identifiers are expected to survive the default policy"

    strict_all = Redactor(override=RedactionStrategy.MASK)
    assert not residual_classes(text, findings, strict_all)


def test_redacting_already_redacted_text_is_stable():
    suite = default_suite()
    text = "SSN 482-19-7734 on file."
    redactor = Redactor(pseudonymizer=Pseudonymizer(KEY))
    once = redactor.apply(text, suite.run(text)).text
    twice = redactor.apply(once, suite.run(once)).text
    assert once == twice


def test_strategy_none_leaves_the_value_untouched():
    text = "Title: Staff Platform Engineer"
    from boundary.detectors import detect_context

    findings = [f for f in detect_context(text) if f.data_class is DataClass.JOB_TITLE]
    assert findings
    out = Redactor().apply(text, findings)
    assert out.text == text
    assert out.count == 0


def test_pseudonymizer_refuses_an_empty_key():
    with pytest.raises(ValueError):
        Pseudonymizer(b"")


def test_quasi_identifier_set_ignores_direct_identifiers():
    suite = default_suite()
    findings = suite.run("SSN 482-19-7734. Zip: 78704.")
    quasi = quasi_identifier_set(findings)
    assert DataClass.SSN not in quasi
    assert DataClass.POSTAL_CODE in quasi


def test_no_detected_value_survives_redaction_in_any_corpus_document():
    """Nothing a detector found comes out of the redactor in cleartext.

    This is the property, and it holds over the shipped corpus and not only over
    one hand-built string, because the two ways it has failed both need
    real documents with several findings in one line to show themselves.

    Mutation check, executed against the redactor itself: applying the spans
    forward instead of in descending start order leaves later spans at stale
    offsets, and the tail of every shifted value survives. Returning the digits
    from `_partial` instead of the masked form forwards a whole card number.
    Both are caught here because this asserts on `Redactor.apply`'s OWN output,
    not on a locally recomputed imitation of it.

    Classes redacted with NONE are excluded: leaving those in place is the
    documented behavior, not a leak.
    """
    corpus = build_corpus()
    redactor = Redactor(Pseudonymizer(KEY))
    checked = 0

    for doc in corpus.store.documents.values():
        findings = default_suite(ROSTER).run(doc.text)
        out = redactor.apply(doc.text, findings).text
        for finding in findings:
            if redactor.strategy_for(finding.data_class) is RedactionStrategy.NONE:
                continue
            raw = doc.text[finding.span.start : finding.span.end]
            assert raw not in out, (
                f"{doc.doc_id}: {finding.data_class.value} value {raw!r} "
                f"survived redaction in {out!r}"
            )
            digits = "".join(ch for ch in raw if ch.isdigit())
            if len(digits) > 4:
                assert digits not in out, (
                    f"{doc.doc_id}: {finding.data_class.value} digits {digits!r} "
                    f"survived redaction in {out!r}"
                )
            checked += 1

    assert checked >= 12, f"only {checked} redactions checked; the corpus shrank"
