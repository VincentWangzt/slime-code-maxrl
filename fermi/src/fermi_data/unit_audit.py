"""Deterministic answer-unit audit for Open Scioly Fermi-Eval questions.

Usage contract
--------------
This module is the only place that classifies whether a Fermi-Eval question
specifies an answer unit, needs no unit, or requires a unit that ``data.js``
does not provide. The stage-one command ``fermi-data audit`` calls
``classify_unit_requirement`` after exact normalized-question deduplication.

The classifier is an ordered, rule-based heuristic: explicit-unit rules run
first, then count/dimensionless rules, then dimensional-without-unit rules and
a conservative fallback. Do not call it while constructing train/val/test
splits. Those splits must consume the persisted
``fermi_eval_decontaminated.parquet`` artifact so the audit is a visible,
reproducible stage boundary.

After modifying this file, rerun ``fermi-data audit`` and then
``fermi-data process``. The CLI stores a SHA-256 of this file in the prepared
Parquet metadata and refuses to consume an artifact produced by different
audit logic.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path


EXPLICIT_UNIT = "explicit_unit_specified"
UNIT_NOT_NEEDED = "unit_not_needed"
UNIT_REQUIRED_BUT_UNSPECIFIED = "unit_required_but_unspecified"
UNIT_CLASSIFICATIONS = (EXPLICIT_UNIT, UNIT_NOT_NEEDED, UNIT_REQUIRED_BUT_UNSPECIFIED)


def audit_logic_sha256() -> str:
    """Fingerprint the exact classifier source used to create an audit artifact."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@dataclass(frozen=True)
class UnitAuditResult:
    classification: str
    specified_unit: str
    reason: str
    confidence: str
    needs_review: bool


_UNIT_ALIASES = (
    r"(?:square|sq\.?)\s+(?:nanometers?|micrometers?|millimeters?|centimeters?|meters?|kilometers?|inches?|feet|foot|yards?|miles?|nm|mm|cm|km|m|ft|in)",
    r"(?:cubic|cu\.?)\s+(?:nanometers?|micrometers?|millimeters?|centimeters?|meters?|kilometers?|inches?|feet|foot|yards?|miles?|nm|mm|cm|km|m|ft|in)",
    r"(?:feet|foot|meters?|kilometers?|inches?|yards?|miles?)\s+per\s+(?:second|minute|hour)",
    r"(?:m|cm|mm|km|ft|in|mi)\s*(?:/|per)\s*(?:s|sec|second|h|hr|hour)",
    r"(?:degrees?\s+)?(?:celsius|fahrenheit|kelvin)",
    r"degrees?\s+(?:c|f)",
    r"(?:kilowatt|megawatt|gigawatt)[ -]?hours?",
    r"electron[ -]?volts?",
    r"planck\s+units?",
    r"astronomical\s+units?",
    r"light[ -]?(?:seconds?|minutes?|hours?|years?)",
    r"light[ -]?(?:feet|foot)",
    r"acre[ -]?feet",
    r"(?:acres?|hectares?)",
    r"pounds?[ -]?force",
    r"(?:femtoseconds?|picoseconds?|nanoseconds?|microseconds?|milliseconds?|deciseconds?|seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|yrs?|decades?|centuries?|millennia|jiff(?:y|ies))",
    r"(?:angstroms?|femtometers?|picometers?|nanometers?|micrometers?|microns?|millimeters?|centimeters?|meters?|kilometers?|megameters?|gigameters?|petameters?|inches?|feet|feets|foot|yards?|miles?|furlongs?|fathoms?|parsecs?)",
    r"(?:nanograms?|micrograms?|milligrams?|grams?|kilograms?|megagrams?|gigagrams?|petagrams?|tonnes?|metric\s+tons?|tons?|pounds?|lbs?|ounces?|oz|troy\s+ounces?|stones?|slugs?|daltons?|carats?)",
    r"(?:microliters?|milliliters?|liters?|litres?|gallons?|quarts?|pints?|cups?|teaspoons?|tablespoons?|barrels?)",
    r"(?:joules?|kilojoules?|megajoules?|gigajoules?|picojoules?|calories?|kilocalories?|kcal|kilowatts?|megawatts?|gigawatts?|watts?|horsepower)",
    r"(?:newtons?|kilonewtons?|millinewtons?|meganewtons?|pascals?|kilopascals?|megapascals?|atmospheres?|"
    r"atm|psi|torr|mmhg|decibels?|db|sieverts?|barns?|g['’]?s)",
    r"(?:(?:(?:19|20)\d{2}\s+)?dollars?|cents?|u\.?s\.?\s+dollars?|usd|euros?|(?:japanese\s+)?yen|yuan|rupees?|francs?|lek|"
    r"pounds?\s+sterling|german\s+marks?|(?:new\s+taiwan|mexican|singapore|ukrainian|zimbabwean)\s+dollars?)",
    r"(?:bits?|bytes?|kilobytes?|megabytes?|gigabytes?|terabytes?|petabytes?)",
    r"(?:hertz|kilohertz|megahertz|gigahertz)",
    r"(?:volts?|millivolts?|kilovolts?|amperes?|amps?|milliamps?|ohms?|coulombs?)",
    r"(?:moles?|molar|percent|percentage|radians?|degrees?|sverdrups?|rpm|btus?|scoville\s+units?|centitones?)",
    r"(?:picas?)",
    r"(?:nanograms?|micrograms?|milligrams?|grams?|kilograms?|liters?|joules?)\s+per\s+(?:second|minute|hour|day|year)",
    r"(?:nanograms?|micrograms?|milligrams?|grams?|kilograms?)\s*/\s*(?:cubic\s+)?(?:meters?|centimeters?|liters?)",
    r"(?:g|kg)\s*/\s*(?:m|cm)\s*(?:\^?\s*3)?",
    r"(?:[fpnumk]?m|μm|cm|mm|km|ft|in)\s*(?:\^?\s*[23]|squared|cubed)",
    r"(?:w|kw|mw|gw)\s*/\s*(?:m|cm)\s*(?:\^?\s*2)?",
    r"(?:m|cm|mm|km|ft|in)\s*/\s*s\s*(?:\^?\s*2)?",
    r"(?:mph|kph|hz|khz|mhz|ghz|kb|mb|gb|tb|kwh)",
    r"(?:nm|μm|um|mm|cm|km|ft|yd|mi|mg|kg|ml|kj|mj|kw|mw|gw|pa|kpa|kv)",
)
_UNIT = "(?:" + "|".join(_UNIT_ALIASES) + ")"

_DIMENSIONAL_TARGET = (
    r"mass|weight|distance|length|height|width|depth|diameter|radius|circumference|area|surface\s+area|"
    r"volume|speed|velocity|time|duration|energy|electricity|power|force|pressure|temperature|density|cost|price|"
    r"money|worth|"
    r"frequency|voltage|current|wavelength|thickness|lifetime|acceleration|momentum|torque|capacitance|"
    r"resistance|resistivity|charge|flux|luminosity|entropy|heat|memory|storage|data|revenue|income|salary|"
    r"sales|expenses?|exports?|gdp|endowment|net\s+worth|market\s+cap|value|concentration|latency|elevation|"
    r"font\s+size|size|age|half[ -]?life|attraction|pull|separation|amplification|intensity|conductivity|"
    r"period|efficiency|difference|damage|loss|perimeter|consumption|volumetric\s+flow|flow|flow\s+rate|rate"
)

_LEXICAL_REQUEST_PATTERNS = (
    re.compile(
        rf"\b(?:answer|express|measure|calculate|report|give)(?:ed|d)?(?:\s+(?:the|your)\s+answer)?\s+in\s+"
        rf"(?P<unit>{_UNIT})(?!\w)",
        re.IGNORECASE,
    ),
    re.compile(rf"(?:\(|\[)\s*(?:in\s+)?(?P<unit>{_UNIT})\s*(?:\)|\])\s*[?.]?\s*$", re.IGNORECASE),
    re.compile(rf"\b(?:in|into)\s+(?P<unit>{_UNIT})(?!\w)(?=\s*[?.]?\s*$)", re.IGNORECASE),
    re.compile(rf"\bhow\s+many\s+(?P<unit>{_UNIT})(?!\w)", re.IGNORECASE),
    re.compile(
        rf"\b(?:{_DIMENSIONAL_TARGET})\b\s*,?\s+(?:(?:measured|expressed)\s+)?in\s+"
        rf"(?P<unit>{_UNIT})(?!\w)",
        re.IGNORECASE,
    ),
    re.compile(rf"\b(?:weigh|cost)\b[^?]{{0,60}}?\bin\s+(?P<unit>{_UNIT})(?!\w)", re.IGNORECASE),
    re.compile(rf"\bhow\s+much\b[^?,]{{0,80}}?,\s+in\s+(?P<unit>{_UNIT})(?!\w)", re.IGNORECASE),
    re.compile(
        rf"\b(?:difference|value|revenue|income|salary|sales|expenses?|gdp|endowment|net\s+worth)\b[^?]{{0,40}}?"
        rf"\bin\s+(?P<unit>{_UNIT})(?!\w)",
        re.IGNORECASE,
    ),
    re.compile(rf"^\s*in\s+(?P<unit>{_UNIT})(?!\w)\s*,", re.IGNORECASE),
    re.compile(rf"\bconvert\b[^?]*?\bto\s+(?P<unit>{_UNIT})(?!\w)", re.IGNORECASE),
    re.compile(
        rf"\b(?:express|find|state)\b[^?]{{0,180}}?\b(?:in|in\s+units?\s+of)\s+(?P<unit>{_UNIT})(?!\w)",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bhow\s+(?:long|far|fast|heavy|hot|cold|tall|deep|high|wide|thick|old|loud|attracted|"
        rf"radioactive|spicy)\b"
        rf"[^?]{{0,100}}?"
        rf"\bin\s+(?P<unit>{_UNIT})(?!\w)",
        re.IGNORECASE,
    ),
)
_DIMENSIONAL_INTERROGATIVE = re.compile(
    r"\bhow\s+(?:long|far|fast|heavy|hot|cold|tall|deep|high|wide|thick|old|big|large|loud|attracted|"
    r"radioactive|spicy)\b",
    re.IGNORECASE,
)
_DIMENSIONAL_TARGET_QUESTION = re.compile(
    rf"(?:\bwhat\s+(?:(?:is|are|was|were|would|will|should|could)\s+)?(?:the|a|an)?[^?]{{0,40}}?|"
    rf"\b(?:estimate|calculate|find|determine)\s+(?:the|a|an)?[^?]{{0,25}}?|"
    rf"^\s*(?:the|a|an)?(?:\s*[A-Za-z-]+){{0,4}}?\s*)\b(?:{_DIMENSIONAL_TARGET})\b",
    re.IGNORECASE,
)
_HOW_MUCH = re.compile(r"\bhow\s+much\b", re.IGNORECASE)
_WEIGH_OR_COST_QUESTION = re.compile(r"\b(?:what\s+does|how\s+much\s+does)\b[^?]{0,100}?\b(?:weigh|cost)\b", re.I)

_GENERIC_TRAILING_UNIT = re.compile(
    r"(?:\b(?:measured\s+in|expressed\s+in|in\s+(?:terms|units)\s+of|as\s+(?:a\s+)?multiple\s+of)|"
    r",\s*(?:in|into)(?!\s+(?:terms|units)\s+of))\s+"
    r"(?P<unit>[A-Za-z°$][A-Za-z0-9°$/*^.'-]*(?:\s+[A-Za-z°$][A-Za-z0-9°$/*^.'-]*){0,8})"
    r"\s*[?.!]?\s*$",
    re.IGNORECASE,
)
_TARGET_KNOWN_UNIT = re.compile(
    rf"\b(?:{_DIMENSIONAL_TARGET})\b[^?]{{0,160}}?"
    rf"(?:\bin\b|\binto\b|\bas\s+(?:a\s+)?multiple\s+of\b|\b(?:measured|expressed)\s+in\b|"
    rf"\bin\s+(?:terms|units)\s+of\b)\s+(?P<unit>{_UNIT})(?!\w)",
    re.IGNORECASE,
)
_TARGET_ADJACENT_KNOWN_UNIT = re.compile(
    rf"\b(?:{_DIMENSIONAL_TARGET})\b\s*,?\s+(?P<unit>{_UNIT})(?!\w)",
    re.IGNORECASE,
)
_TARGET_PER_KNOWN_UNIT = re.compile(
    rf"\b(?:{_DIMENSIONAL_TARGET})\b[^?]{{0,120}}?\bper\s+(?P<unit>{_UNIT})(?!\w)",
    re.IGNORECASE,
)
_RATE_OF_UNIT = re.compile(
    rf"\brate\s+of\s+(?P<unit>{_UNIT})(?!\w)(?:\s+per\s+[^?]+)?",
    re.IGNORECASE,
)
_TARGET_OPEN_TRAILING_UNIT = re.compile(
    rf"\b(?:{_DIMENSIONAL_TARGET})\b[^?]{{0,160}}?"
    r"(?:\bas\s+(?:a\s+)?multiple\s+of\b|\bmeasured\s+(?:in\s+)?units?\s+of\b|"
    r"\b(?:measured|expressed)\s+in\b|"
    r"\bin\s+(?:terms|units)\s+of\b)\s+"
    r"(?P<unit>[A-Za-z°$][A-Za-z0-9°$/*^.'-]*(?:\s+[A-Za-z°$][A-Za-z0-9°$/*^.'-]*){0,15})"
    r"\s*[?.!]?\s*$",
    re.IGNORECASE,
)
_TARGET_RAW_TRAILING_UNIT = re.compile(
    rf"\b(?:{_DIMENSIONAL_TARGET})\b[^?]{{0,180}}?\bin\s+"
    r"(?P<unit>[A-Za-z°$][A-Za-z0-9°$/*^.'-]*(?:\s+[A-Za-z°$][A-Za-z0-9°$/*^.'-]*){0,5})"
    r"\s*[?.!]?\s*$",
    re.IGNORECASE,
)
_INTERROGATIVE_OPEN_TRAILING_UNIT = re.compile(
    r"\bhow\s+(?:long|far|fast|heavy|hot|cold|tall|deep|high|wide|thick|old|big|large|loud|attracted|radioactive)\b"
    r"[^?]{0,180}?\bin\s+"
    r"(?P<unit>[A-Za-z°$][A-Za-z0-9°$/*^.'-]*(?:\s+[A-Za-z°$][A-Za-z0-9°$/*^.'-]*){0,4})"
    r"\s*[?.!]?\s*$",
    re.IGNORECASE,
)
_OPEN_PREFIX_UNIT = re.compile(
    rf"(?:^|[,!?\.]\s*)in\s+(?P<unit>{_UNIT}|[A-Za-z°$][A-Za-z0-9°$/*^.'-]*(?:\s+[A-Za-z°$][A-Za-z0-9°$/*^.'-]*){{0,3}})"
    rf"\s*,\s*[^?]{{0,100}}?(?:\b(?:{_DIMENSIONAL_TARGET})\b|"
    r"\bhow\s+(?:long|far|fast|heavy|hot|cold|tall|deep|high|wide|thick|old|big|large|loud|attracted)\b)",
    re.IGNORECASE,
)
_REPORT_OPEN_UNIT = re.compile(
    r"\breport\s+(?:the|your)\s+answer\s+in\s+"
    r"(?P<unit>[A-Za-z°$][A-Za-z0-9°$/*^.'-]*(?:\s+[A-Za-z°$][A-Za-z0-9°$/*^.'-]*){0,6})"
    r"(?=\s*\(|\s*[?.!]?$)",
    re.IGNORECASE,
)
_GENERIC_INLINE_UNIT = re.compile(
    rf"\b(?:{_DIMENSIONAL_TARGET})\b\s*,?\s+in\s+"
    r"(?P<unit>[A-Za-z°$][A-Za-z0-9°$/*^.'-]*(?:\s+[A-Za-z°$][A-Za-z0-9°$/*^.'-]*){0,2})"
    r"(?=\s+(?:of|for|would|will|does|do|did|is|are|was|were|needed|required|produced|generated)\b|\s*[?.!,])",
    re.IGNORECASE,
)
_GENERIC_INTERROGATIVE_UNIT = re.compile(
    r"\bhow\s+(?:long|far|fast|heavy|hot|cold|tall|deep|high|wide|thick|old|big|large)\s+in\s+"
    r"(?P<unit>[A-Za-z°$][A-Za-z0-9°$/*^.'-]*(?:\s+[A-Za-z°$][A-Za-z0-9°$/*^.'-]*){0,2})"
    r"(?=\s+(?:would|will|does|do|did|is|are|was|were|can|could|should)\b|\s*[?.!,])",
    re.IGNORECASE,
)

_PERCENT_REQUEST = re.compile(
    r"\b(?:what|which|approximately\s+what|how\s+many)\s+(?:percent|percentage)\b|"
    r"\bpercent(?:age)?\s+(?:increase|decrease|change|difference)\b|%\s*[?.]?\s*$",
    re.IGNORECASE,
)
_OPEN_COMMAND_UNIT = re.compile(
    r"(?:\bexpress\b[^?,.!]{0,220}?\b(?:in|units?\s+of)|\bconvert\b[^?,.!]{0,220}?\bto)\s+"
    r"(?P<unit>[^?,.!]{1,160}?)\s*[?.]?\s*$",
    re.IGNORECASE,
)
_ANSWER_MARKER = re.compile(r"\[(?P<unit>s|sec|seconds?|\$|USD)\]\s*[?.]*\s*$", re.IGNORECASE)
_UNIT_FULLMATCH = re.compile(rf"{_UNIT}", re.IGNORECASE)
_BRACKETED_OUTPUT_CANDIDATE = re.compile(
    r"(?:\(|\[)\s*(?P<unit>[^\])]{1,40})\s*(?:\)|\])\s*[?.]?\s*$",
    re.IGNORECASE,
)
_SQUARE_UNIT_ANYWHERE = re.compile(r"\[\s*(?P<unit>[^\]]{1,40})\s*\]", re.IGNORECASE)
_CURRENCY_MARKER_ANYWHERE = re.compile(
    r"(?:\(|\[)\s*(?:in\s+)?(?P<unit>(?:(?:19|20)\d{2}\s+)?(?:USD|U\.?S\.? ?\s*\$|"
    r"German\s+Marks?))\s*(?:\)|\])",
    re.IGNORECASE,
)
_SI_OR_USD_MARKER = re.compile(r"\[\s*Use\s+SI\s+Units\s*/\s*2017\s+USD\s*\]", re.IGNORECASE)
_WHAT_YEAR = re.compile(r"\bwhat\s+year\b", re.IGNORECASE)
_LEADING_UNIT_FRAGMENT = re.compile(
    rf"^\s*(?:the\s+)?(?:total\s+)?(?P<unit>{_UNIT})(?!\w)\s+(?:of|in|since|paid|raised|spent|used|"
    r"watched|burned|consumed|produced|created|generated|per|on|to\s+cover)\b",
    re.IGNORECASE,
)
_DIMENSIONLESS_REQUESTS = (
    (
        re.compile(r"\b(?:ratio|fraction|proportion|probability|chance|chances|odds)\b", re.I),
        "ratio, fraction, or probability",
    ),
    (re.compile(r"\b(?:acceptance|success|failure|conversion|survival)\s+rate\b", re.I), "dimensionless rate"),
    (
        re.compile(
            r"\b(?:how\s+many\s+times|by\s+what\s+factor|multiplicative\s+factor|orders?\s+of\s+magnitude)\b",
            re.I,
        ),
        "dimensionless factor",
    ),
    (
        re.compile(
            r"\b(?:compared\s+to|relative\s+to|multiple\s+increase|covered\s+by|divide\b.+\bby|"
            r"how\s+much\s+more\s+likely|how\s+much\s+of\b[^?]*\bcover)\b",
            re.I,
        ),
        "dimensionless comparison",
    ),
    (re.compile(r"\b(?:number|count|population)\s+of\b", re.I), "count"),
    (re.compile(r"\b(?:ways?|arrangements?|permutations?|combinations?)\b", re.I), "combinatorial count"),
    (
        re.compile(
            r"\b(?:sum|product|factorial|digit|integer|prime|compute|evaluate|solve\s+for|sequence|"
            r"natural\s+log|logarithm|(?:raised\s+)?to\s+the\s+power|divided\s+by|quotient|"
            r"sine|cosine|tangent|sin\s*\(|cos\s*\(|tan\s*\(|fibonacci|choose|cube\s*root|sqrt|"
            r"triangular\s+number|pentagonal\s+number|tetration|pentation|hexation|"
            r"\d+(?:st|nd|rd|th)\s+power|value\s+of\s+(?:the\s+expression|e|pi|π|\d)|"
            r"cardinality|mode\s+of|f\s*\(|c\s*\(|coefficient\s+of\s+(?:kinetic\s+)?friction|\bpH\b)",
            re.I,
        ),
        "pure numeric or mathematical value",
    ),
)
_HOW_MANY = re.compile(r"\bhow\s+many\b", re.IGNORECASE)
_HOW_MANY_TYPO = re.compile(r"\b(?:hwo\s+many|howmany)\b", re.IGNORECASE)
_REMAINING_FRACTION = re.compile(r"\bhow\s+much\s+of\b[^?]*\b(?:left|remain(?:s|ing)?)\b", re.IGNORECASE)
_POPULATION_COUNT = re.compile(r"\bpopulation\b(?!\s+density)", re.IGNORECASE)
_OTHER_DIMENSIONLESS = re.compile(
    r"\b(?:team\s+number|phone\s+number|page\s+number|rank(?:ing)?|place|score|capacity|occupancy|"
    r"viewership|(?:user|pixel)\s+count|unique\s+hands?|index|equilibrium\s+constant|\[#\]|"
    r"scale\s+(?:from|of)\s+\d)\b",
    re.IGNORECASE,
)
_PURE_EXPRESSION = re.compile(
    r"^\s*(?:(?:find|compute|evaluate)\s+)?(?:what\s+is\s+)?[0-9()^+*/!.\s-]+[?.]?\s*$",
    re.IGNORECASE,
)
_SHORT_MATH_EXPRESSION = re.compile(r"(?:\d|\bpi\b|π|\be\b)\s*(?:\^|×|\bx\b|\*|\+|/|!|−)", re.IGNORECASE)
_COUNT_NOUNS = re.compile(
    r"\b(?:total\s+)?(?:people|persons?|humans?|students?|birds?|animals?|trees?|cells?|atoms?|molecules?|"
    r"electrons?|protons?|neutrons?|photons?|grains?|drops?|words?|letters?|characters?|digits?|objects?|"
    r"items?|events?|games?|books?|pages?|cars?|vehicles?|planes?|ships?|buildings?|houses?|stars?|planets?|"
    r"transistors?|species|skulls?|rabbits?|elephants?|pigeons?|baguettes?|farmers?|papers?|lines?|code|"
    r"hands?|phones?|cellphones?|horses?|"
    r"breaths?|candles?|prisoners?|muons?|videos?|diamonds?|carats?|"
    r"coins?|pennies|dimes?|nickels?|quarters?|bottles?|cans?|balls?|sheets?|hairs?|ways?)\b",
    re.IGNORECASE,
)
_AMOUNT_OF_COUNT = re.compile(r"\bamount\s+of\s+" + _COUNT_NOUNS.pattern.removeprefix(r"\b"), re.IGNORECASE)
_REFERENCE_DEPENDENCY = re.compile(
    r"\b(?:question|answer|result|value|number)\s+(?:above|below|before|previous|preceding|following|earlier)\b|"
    r"\b(?:previous|preceding|following|earlier|next)\s+question\b|\bquestion\s*#?\s*\d+\b",
    re.IGNORECASE,
)


def classify_unit_requirement(question: str) -> UnitAuditResult:
    """Classify whether an exponent label has a recoverable answer scale from the question text."""
    normalized = " ".join(re.sub(r"[\u200b-\u200d\ufeff]", " ", question).split())
    if not normalized:
        raise ValueError("Cannot audit an empty question")

    explicit = _explicit_unit_request(normalized)
    if explicit is not None:
        specified_unit, reason, confidence = explicit
        return UnitAuditResult(
            classification=EXPLICIT_UNIT,
            specified_unit=specified_unit,
            reason=reason,
            confidence=confidence,
            needs_review=confidence != "high" or bool(_REFERENCE_DEPENDENCY.search(normalized)),
        )

    if _PURE_EXPRESSION.fullmatch(normalized) or (
        len(normalized) <= 160
        and _SHORT_MATH_EXPRESSION.search(normalized)
        and not _DIMENSIONAL_TARGET_QUESTION.search(normalized)
    ):
        return UnitAuditResult(
            classification=UNIT_NOT_NEEDED,
            specified_unit="",
            reason="Answer is a pure numeric or mathematical value",
            confidence="high",
            needs_review=bool(_REFERENCE_DEPENDENCY.search(normalized)),
        )

    if re.fullmatch(r"(?i).*\bwhat\s+is\s+[a-z]\s*[?.]?", normalized):
        return UnitAuditResult(
            classification=UNIT_NOT_NEEDED,
            specified_unit="",
            reason="Answer is a pure numeric variable",
            confidence="medium",
            needs_review=True,
        )

    for pattern, label in _DIMENSIONLESS_REQUESTS:
        if pattern.search(normalized):
            return UnitAuditResult(
                classification=UNIT_NOT_NEEDED,
                specified_unit="",
                reason=f"Answer is an inherent {label}",
                confidence="high",
                needs_review=bool(_REFERENCE_DEPENDENCY.search(normalized)),
            )

    if _REMAINING_FRACTION.search(normalized):
        return UnitAuditResult(
            classification=UNIT_NOT_NEEDED,
            specified_unit="",
            reason="Answer is an inherent remaining fraction",
            confidence="medium",
            needs_review=True,
        )

    count_fragment = bool(
        _COUNT_NOUNS.match(normalized)
        or (normalized.casefold().startswith("how about") and _COUNT_NOUNS.search(normalized))
        or (normalized.casefold().startswith("amount of") and _COUNT_NOUNS.search(normalized))
        or (
            len(normalized.split()) <= 16
            and _COUNT_NOUNS.search(normalized)
            and not _DIMENSIONAL_TARGET_QUESTION.search(normalized)
        )
    )
    if (
        _POPULATION_COUNT.search(normalized)
        or _OTHER_DIMENSIONLESS.search(normalized)
        or _AMOUNT_OF_COUNT.search(normalized)
        or count_fragment
    ):
        return UnitAuditResult(
            classification=UNIT_NOT_NEEDED,
            specified_unit="",
            reason="Population is an inherent count",
            confidence="high",
            needs_review=bool(_REFERENCE_DEPENDENCY.search(normalized)),
        )

    if _HOW_MANY.search(normalized) or _HOW_MANY_TYPO.search(normalized):
        confidence = "high" if _COUNT_NOUNS.search(normalized) else "medium"
        return UnitAuditResult(
            classification=UNIT_NOT_NEEDED,
            specified_unit="",
            reason="'How many' requests a count rather than a dimensional magnitude",
            confidence=confidence,
            needs_review=confidence != "high" or bool(_REFERENCE_DEPENDENCY.search(normalized)),
        )

    if _DIMENSIONAL_INTERROGATIVE.search(normalized):
        return UnitAuditResult(
            classification=UNIT_REQUIRED_BUT_UNSPECIFIED,
            specified_unit="",
            reason="Dimensional interrogative has no requested output unit",
            confidence="high",
            needs_review=True,
        )

    if _DIMENSIONAL_TARGET_QUESTION.search(normalized):
        return UnitAuditResult(
            classification=UNIT_REQUIRED_BUT_UNSPECIFIED,
            specified_unit="",
            reason="Dimensional target quantity has no requested output unit",
            confidence="high",
            needs_review=True,
        )

    if _HOW_MUCH.search(normalized):
        return UnitAuditResult(
            classification=UNIT_REQUIRED_BUT_UNSPECIFIED,
            specified_unit="",
            reason="'How much' requests a magnitude but supplies no output unit",
            confidence="medium",
            needs_review=True,
        )

    return UnitAuditResult(
        classification=UNIT_REQUIRED_BUT_UNSPECIFIED,
        specified_unit="",
        reason="Question supplies neither an output unit nor clear count/dimensionless semantics",
        confidence="low",
        needs_review=True,
    )


def requested_unit(question: str) -> str | None:
    """Return a detected unit explicitly requested as the output basis."""
    explicit = _explicit_unit_request(" ".join(re.sub(r"[\u200b-\u200d\ufeff]", " ", question).split()))
    return explicit[0] if explicit is not None else None


def unit_audit_status(question: str) -> tuple[str, str]:
    """Return the three-way classification and any explicitly requested unit."""
    result = classify_unit_requirement(question)
    return result.classification, result.specified_unit


def _explicit_unit_request(question: str) -> tuple[str, str, str] | None:
    if _PERCENT_REQUEST.search(question):
        return "percent", "Question explicitly requests a percentage scale", "high"
    if re.search(r"\b(?:as|expressed\s+as)\s+a\s+percentage\b", question, re.IGNORECASE):
        return "percent", "Question explicitly requests a percentage scale", "high"
    if re.search(r"\bdaily\s+value\s*%", question, re.IGNORECASE):
        return "percent", "Question explicitly requests a percentage scale", "high"
    if _WHAT_YEAR.search(question):
        return "year", "Question explicitly requests a calendar year", "high"
    answer_marker = _ANSWER_MARKER.search(question)
    if answer_marker:
        unit = answer_marker.group("unit")
        return ("dollars" if unit == "$" else unit), "Question includes an explicit answer-unit marker", "high"
    bracketed_unit = _BRACKETED_OUTPUT_CANDIDATE.search(question)
    if bracketed_unit and _is_bracketed_unit(bracketed_unit.group("unit")):
        return (
            _clean_unit(re.sub(r"^in\s+", "", bracketed_unit.group("unit"), flags=re.IGNORECASE)),
            "Question includes an explicit bracketed answer unit",
            "high",
        )
    square_units = [
        match
        for match in _SQUARE_UNIT_ANYWHERE.finditer(question)
        if _is_bracketed_unit(match.group("unit")) or _is_custom_square_unit(match.group("unit"))
    ]
    dimensional = bool(
        _DIMENSIONAL_INTERROGATIVE.search(question)
        or _DIMENSIONAL_TARGET_QUESTION.search(question)
        or _HOW_MUCH.search(question)
        or _WEIGH_OR_COST_QUESTION.search(question)
    )
    if square_units and dimensional:
        return (
            _clean_unit(re.sub(r"^in\s+", "", square_units[-1].group("unit"), flags=re.IGNORECASE)),
            "Question includes an explicit bracketed answer unit",
            "high",
        )
    currency_marker = _CURRENCY_MARKER_ANYWHERE.search(question)
    if currency_marker and re.search(
        r"\b(?:cost|price|revenue|income|salary|sales|expenses?|gdp|endowment|net\s+worth|market\s+cap|value)\b",
        question,
        re.IGNORECASE,
    ):
        return (
            _clean_unit(currency_marker.group("unit")),
            "Question includes an explicit currency answer marker",
            "high",
        )
    currency_value = re.search(r"\b(?P<unit>USD|dollars?)\s+value\b", question, re.IGNORECASE)
    if currency_value:
        return (
            currency_value.group("unit"),
            "Question names the currency of the requested value",
            "high",
        )
    leading_unit = _LEADING_UNIT_FRAGMENT.search(question)
    if leading_unit:
        return _clean_unit(leading_unit.group("unit")), "Question fragment begins with the requested unit", "medium"
    report_unit = _REPORT_OPEN_UNIT.search(question)
    if report_unit:
        return (
            _clean_unit(report_unit.group("unit")),
            "Question explicitly directs the answer to be reported in named units",
            "medium",
        )
    for pattern in _LEXICAL_REQUEST_PATTERNS:
        matches = list(pattern.finditer(question))
        if matches:
            return _clean_unit(matches[-1].group("unit")), "Question explicitly names the requested output unit", "high"

    target_known_unit = _TARGET_KNOWN_UNIT.search(question)
    if target_known_unit and dimensional:
        return (
            _clean_unit(target_known_unit.group("unit")),
            "The named dimensional target is expressed in an explicit unit",
            "high",
        )
    for pattern in (_TARGET_ADJACENT_KNOWN_UNIT, _TARGET_PER_KNOWN_UNIT, _RATE_OF_UNIT):
        target_unit = pattern.search(question)
        if target_unit and dimensional:
            return (
                _clean_unit(target_unit.group("unit")),
                "The named dimensional target is expressed in an explicit unit",
                "high",
            )

    explicit_target_unit = re.search(r"\bwhat\b[^?]{0,80}?\b(?P<unit>horsepower)\b", question, re.IGNORECASE)
    if explicit_target_unit:
        return (
            explicit_target_unit.group("unit"),
            "Question names the unit as the answer quantity",
            "medium",
        )

    open_command = _OPEN_COMMAND_UNIT.search(question)
    if open_command:
        return (
            _clean_unit(open_command.group("unit")),
            "Question explicitly requests a conversion or expression in named units",
            "medium",
        )
    state_unit = re.search(
        r"(?:^|[.!?]\s+)state\b[^?]{0,180}?\bin\s+"
        r"(?P<unit>[A-Za-z][A-Za-z0-9/*^.'-]*(?:\s+[A-Za-z][A-Za-z0-9/*^.'-]*){0,5})(?=,|[?.]|$)",
        question,
        re.IGNORECASE,
    )
    if state_unit:
        return (
            _clean_unit(state_unit.group("unit")),
            "Question explicitly directs the answer to be stated in named units",
            "medium",
        )

    prefix_unit = _OPEN_PREFIX_UNIT.search(question)
    if prefix_unit:
        return (
            _clean_unit(prefix_unit.group("unit")),
            "Question places the requested output unit before the dimensional target",
            "medium",
        )

    dimensionless_rate = re.search(r"\b(?:acceptance|success|failure|conversion|survival)\s+rate\b", question, re.I)
    if (
        _SI_OR_USD_MARKER.search(question)
        and (_DIMENSIONAL_TARGET_QUESTION.search(question) or _DIMENSIONAL_INTERROGATIVE.search(question))
        and not dimensionless_rate
    ):
        financial = re.search(r"\b(?:cost|price|revenue|income|salary|gdp|endowment|net\s+worth|value)\b", question, re.I)
        return (
            "2017 USD" if financial else "SI unit for target quantity",
            "Question supplies an answer-system marker that fixes the output scale",
            "medium",
        )

    if dimensional:
        for pattern in (
            _GENERIC_INTERROGATIVE_UNIT,
            _GENERIC_INLINE_UNIT,
            _TARGET_OPEN_TRAILING_UNIT,
            _GENERIC_TRAILING_UNIT,
            _INTERROGATIVE_OPEN_TRAILING_UNIT,
            _TARGET_RAW_TRAILING_UNIT,
        ):
            if pattern is _TARGET_RAW_TRAILING_UNIT and (
                _HOW_MANY.search(question)
                or any(candidate.search(question) for candidate, _ in _DIMENSIONLESS_REQUESTS)
            ):
                continue
            matches = list(pattern.finditer(question))
            if matches:
                unit = _clean_unit(matches[-1].group("unit"))
                raw_match = matches[-1].group(0).lstrip()
                if _looks_contextual(unit) and (raw_match.startswith(",") or pattern is _TARGET_RAW_TRAILING_UNIT):
                    continue
                if pattern is _TARGET_RAW_TRAILING_UNIT and re.search(
                    r"\b(?:contained|located|living|found|based|occurred|produced|generated)\s+in\s+[^?]+$",
                    raw_match,
                    re.IGNORECASE,
                ):
                    continue
                return (
                    unit,
                    "Dimensional question explicitly names a nonstandard or open-vocabulary output unit",
                    "medium",
                )
    return None


def _clean_unit(unit: str) -> str:
    words = unit.strip(" \t\r\n,.;:!?()[]").split()
    while words and words[-1].casefold() in {"are", "is", "was", "were", "would", "will"}:
        words.pop()
    return " ".join(words)


def _looks_contextual(unit: str) -> bool:
    normalized = unit.casefold().strip()
    return normalized.startswith(("a ", "an ", "the ", "this ", "that ", "his ", "her ", "their ", "our ", "your "))


def _is_bracketed_unit(unit: str) -> bool:
    normalized = re.sub(r"^in\s+", "", _clean_unit(unit), flags=re.IGNORECASE)
    if normalized.casefold() in {"s", "m", "l", "j", "n", "w", "v", "a", "g", "$", "us $", "u.s. $"}:
        return True
    if "/" in normalized and len(normalized.split()) <= 5:
        return True
    numerator = re.split(r"\s*/\s*", normalized, maxsplit=1)[0]
    return _UNIT_FULLMATCH.fullmatch(numerator) is not None


def _is_custom_square_unit(unit: str) -> bool:
    normalized = _clean_unit(unit)
    if len(normalized.split()) > 6 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9/*^.' -]*", normalized):
        return False
    return not re.match(
        r"(?i)^(?:use|hint|difficulty|previous|question|go\s+with|for\s+reference|assume|given)\b",
        normalized,
    )
