import copy
import io
import json
import math
import os
import re
import time
import unicodedata
import wave
import datetime as dt
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st


# ======================================================================
# CONFIGURATION
# ======================================================================

APP_TITLE = "Global Code-Switched Speech AI Review Console"
APP_SUBTITLE = "Consent-aware multilingual transcription, confidence review, diarisation, and evaluation demo"

CONFIG: Dict[str, Any] = {
    "confidence": {
        "word_low_threshold": 0.75,
        "auto_accept_min": 0.90,
        "flag_avg_confidence_below": 0.82,
    },
    "vad": {
        "sample_rate": 16000,
        "max_segment_s": 30.0,
        "silence_threshold": 0.008,
        "min_speech_s": 0.20,
        "pad_s": 0.05,
        "merge_gap_s": 0.25,
        "snr_threshold_db": 12.0,
    },
    "retention": {
        "audio_days": 90,
        "transcript_days": 365,
    },
    "training": {
        "base_checkpoint": "Whisper-style multilingual encoder-decoder",
        "learning_rate": 1e-5,
        "steps": 3000,
        "speed_min": 0.90,
        "speed_max": 1.10,
        "pitch_change": False,
    },
    "evaluation": {
        "held_out_calls": 200,
        "second_transcriber_rate": 0.15,
        "targets": {
            "overall_wer_max": 0.15,
            "code_switched_wer_max": 0.25,
            "group_gap_points_max": 5.0,
            "low_resource_wer_max": 0.20,
        },
    },
    "privacy": {
        "speaker_codes_instead_of_names": True,
        "generic_labels_without_metadata": True,
        "legal_review_before_deployment": True,
    },
}

PUNCT_MAP = {
    "،": ",",
    "؛": ";",
    "؟": "?",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
    "。": ".",
    "،": ",",
}

URDU_FAMILY = {"Urdu", "Roman Urdu"}


class ConsentError(Exception):
    """Raised when transcription is requested without valid recorded consent."""


# ======================================================================
# GLOBAL LANGUAGE CATALOG
# ======================================================================

LANGUAGE_CATALOG: List[Dict[str, Any]] = [
    {"name": "English", "iso": "en", "scripts": ["Latin"], "region": "Global", "speakers_millions": 1450, "rtl": False, "low_resource": False},
    {"name": "Mandarin Chinese", "iso": "zh", "scripts": ["Han", "Pinyin"], "region": "China / Global", "speakers_millions": 1140, "rtl": False, "low_resource": False},
    {"name": "Hindi", "iso": "hi", "scripts": ["Devanagari", "Latin"], "region": "South Asia", "speakers_millions": 610, "rtl": False, "low_resource": False},
    {"name": "Spanish", "iso": "es", "scripts": ["Latin"], "region": "Europe / Americas", "speakers_millions": 560, "rtl": False, "low_resource": False},
    {"name": "French", "iso": "fr", "scripts": ["Latin"], "region": "Europe / Africa", "speakers_millions": 310, "rtl": False, "low_resource": False},
    {"name": "Arabic", "iso": "ar", "scripts": ["Arabic", "Latin"], "region": "MENA", "speakers_millions": 274, "rtl": True, "low_resource": False},
    {"name": "Bengali", "iso": "bn", "scripts": ["Bengali", "Latin"], "region": "South Asia", "speakers_millions": 273, "rtl": False, "low_resource": True},
    {"name": "Portuguese", "iso": "pt", "scripts": ["Latin"], "region": "Europe / Americas / Africa", "speakers_millions": 258, "rtl": False, "low_resource": False},
    {"name": "Russian", "iso": "ru", "scripts": ["Cyrillic", "Latin"], "region": "Eastern Europe / North Asia", "speakers_millions": 255, "rtl": False, "low_resource": False},
    {"name": "Urdu", "iso": "ur", "scripts": ["Perso-Arabic", "Latin"], "region": "South Asia", "speakers_millions": 232, "rtl": True, "low_resource": False},
    {"name": "Roman Urdu", "iso": "ur-Latn", "scripts": ["Latin"], "region": "South Asia", "speakers_millions": 232, "rtl": False, "low_resource": True},
    {"name": "Indonesian", "iso": "id", "scripts": ["Latin"], "region": "Southeast Asia", "speakers_millions": 199, "rtl": False, "low_resource": False},
    {"name": "German", "iso": "de", "scripts": ["Latin"], "region": "Europe", "speakers_millions": 135, "rtl": False, "low_resource": False},
    {"name": "Persian", "iso": "fa", "scripts": ["Perso-Arabic", "Latin"], "region": "West Asia", "speakers_millions": 130, "rtl": True, "low_resource": True},
    {"name": "Punjabi", "iso": "pa", "scripts": ["Gurmukhi", "Shahmukhi", "Latin"], "region": "South Asia", "speakers_millions": 125, "rtl": False, "low_resource": True},
    {"name": "Japanese", "iso": "ja", "scripts": ["Kanji/Kana", "Latin"], "region": "East Asia", "speakers_millions": 125, "rtl": False, "low_resource": False},
    {"name": "Marathi", "iso": "mr", "scripts": ["Devanagari", "Latin"], "region": "South Asia", "speakers_millions": 99, "rtl": False, "low_resource": True},
    {"name": "Telugu", "iso": "te", "scripts": ["Telugu", "Latin"], "region": "South Asia", "speakers_millions": 96, "rtl": False, "low_resource": True},
    {"name": "Vietnamese", "iso": "vi", "scripts": ["Latin"], "region": "Southeast Asia", "speakers_millions": 85, "rtl": False, "low_resource": True},
    {"name": "Swahili", "iso": "sw", "scripts": ["Latin"], "region": "East Africa", "speakers_millions": 87, "rtl": False, "low_resource": True},
    {"name": "Tamil", "iso": "ta", "scripts": ["Tamil", "Latin"], "region": "South Asia", "speakers_millions": 87, "rtl": False, "low_resource": True},
    {"name": "Turkish", "iso": "tr", "scripts": ["Latin"], "region": "West Asia / Europe", "speakers_millions": 84, "rtl": False, "low_resource": False},
    {"name": "Filipino", "iso": "fil", "scripts": ["Latin"], "region": "Southeast Asia", "speakers_millions": 83, "rtl": False, "low_resource": True},
    {"name": "Korean", "iso": "ko", "scripts": ["Hangul", "Latin"], "region": "East Asia", "speakers_millions": 82, "rtl": False, "low_resource": False},
    {"name": "Italian", "iso": "it", "scripts": ["Latin"], "region": "Europe", "speakers_millions": 68, "rtl": False, "low_resource": False},
    {"name": "Gujarati", "iso": "gu", "scripts": ["Gujarati", "Latin"], "region": "South Asia", "speakers_millions": 62, "rtl": False, "low_resource": True},
    {"name": "Thai", "iso": "th", "scripts": ["Thai", "Latin"], "region": "Southeast Asia", "speakers_millions": 61, "rtl": False, "low_resource": True},
    {"name": "Kannada", "iso": "kn", "scripts": ["Kannada", "Latin"], "region": "South Asia", "speakers_millions": 59, "rtl": False, "low_resource": True},
    {"name": "Polish", "iso": "pl", "scripts": ["Latin"], "region": "Europe", "speakers_millions": 45, "rtl": False, "low_resource": True},
    {"name": "Malayalam", "iso": "ml", "scripts": ["Malayalam", "Latin"], "region": "South Asia", "speakers_millions": 38, "rtl": False, "low_resource": True},
    {"name": "Odia", "iso": "or", "scripts": ["Odia", "Latin"], "region": "South Asia", "speakers_millions": 38, "rtl": False, "low_resource": True},
]

LANGUAGE_NAMES = [x["name"] for x in LANGUAGE_CATALOG]
LANG_BY_NAME = {x["name"]: x for x in LANGUAGE_CATALOG}

DEFAULT_VISIBLE_LANGUAGES = [
    "English",
    "Mandarin Chinese",
    "Hindi",
    "Spanish",
    "French",
    "Arabic",
    "Urdu",
    "Roman Urdu",
    "Bengali",
    "Russian",
    "Japanese",
    "Korean",
]


# ======================================================================
# TOKEN LIBRARY FOR SIMULATED MULTILINGUAL SPEECH
# ======================================================================

TOKEN_LIBRARY: Dict[str, List[Tuple[str, str, str, float]]] = {
    "English": [
        ("hello", "hello", "hello", 0.97),
        ("account", "account", "account", 0.95),
        ("balance", "balance", "balance", 0.93),
        ("payment", "payment", "payment", 0.91),
        ("problem", "problem", "problem", 0.89),
        ("help", "help", "help", 0.94),
    ],
    "Mandarin Chinese": [
        ("你好", "nihao", "hello", 0.94),
        ("账户", "zhanghu", "account", 0.91),
        ("余额", "yue", "balance", 0.89),
        ("付款", "fukuan", "payment", 0.88),
        ("问题", "wenti", "problem", 0.90),
        ("帮助", "bangzhu", "help", 0.92),
    ],
    "Hindi": [
        ("नमस्ते", "namaste", "hello", 0.96),
        ("खाता", "khata", "account", 0.93),
        ("बैलेंस", "balance", "balance", 0.90),
        ("भुगतान", "bhugtan", "payment", 0.89),
        ("समस्या", "samasya", "problem", 0.91),
        ("मदद", "madad", "help", 0.94),
    ],
    "Spanish": [
        ("hola", "hola", "hello", 0.97),
        ("cuenta", "cuenta", "account", 0.95),
        ("saldo", "saldo", "balance", 0.93),
        ("pago", "pago", "payment", 0.92),
        ("problema", "problema", "problem", 0.91),
        ("ayuda", "ayuda", "help", 0.95),
    ],
    "French": [
        ("bonjour", "bonjour", "hello", 0.96),
        ("compte", "compte", "account", 0.94),
        ("solde", "solde", "balance", 0.92),
        ("paiement", "paiement", "payment", 0.91),
        ("problème", "probleme", "problem", 0.90),
        ("aide", "aide", "help", 0.94),
    ],
    "Arabic": [
        ("مرحبا", "marhaban", "hello", 0.95),
        ("حساب", "hisab", "account", 0.93),
        ("رصيد", "rasid", "balance", 0.91),
        ("دفعة", "dafaa", "payment", 0.90),
        ("مشكلة", "mushkila", "problem", 0.90),
        ("مساعدة", "masaada", "help", 0.93),
    ],
    "Bengali": [
        ("হ্যালো", "halo", "hello", 0.95),
        ("অ্যাকাউন্ট", "account", "account", 0.92),
        ("ব্যালেন্স", "balens", "balance", 0.90),
        ("পেমেন্ট", "payment", "payment", 0.89),
        ("সমস্যা", "somossa", "problem", 0.90),
        ("সাহায্য", "sahajjo", "help", 0.92),
    ],
    "Portuguese": [
        ("olá", "ola", "hello", 0.96),
        ("conta", "conta", "account", 0.94),
        ("saldo", "saldo", "balance", 0.92),
        ("pagamento", "pagamento", "payment", 0.91),
        ("problema", "problema", "problem", 0.90),
        ("ajuda", "ajuda", "help", 0.94),
    ],
    "Russian": [
        ("здравствуйте", "zdrastvuyte", "hello", 0.94),
        ("счёт", "schet", "account", 0.91),
        ("баланс", "balans", "balance", 0.90),
        ("платёж", "platezh", "payment", 0.89),
        ("проблема", "problema", "problem", 0.90),
        ("помощь", "pomoshch", "help", 0.92),
    ],
    "Urdu": [
        ("السلام علیکم", "assalam o alaikum", "hello", 0.96),
        ("کھاتہ", "khata", "account", 0.93),
        ("بیلنس", "balance", "balance", 0.90),
        ("ادائیگی", "adaigi", "payment", 0.89),
        ("مسئلہ", "masla", "problem", 0.91),
        ("مدد", "madad", "help", 0.94),
    ],
    "Roman Urdu": [
        ("assalam o alaikum", "assalam o alaikum", "hello", 0.95),
        ("khata", "khata", "account", 0.92),
        ("balance", "balance", "balance", 0.90),
        ("adaigi", "adaigi", "payment", 0.89),
        ("masla", "masla", "problem", 0.90),
        ("madad", "madad", "help", 0.93),
    ],
    "Indonesian": [
        ("halo", "halo", "hello", 0.96),
        ("akun", "akun", "account", 0.94),
        ("saldo", "saldo", "balance", 0.92),
        ("pembayaran", "pembayaran", "payment", 0.90),
        ("masalah", "masalah", "problem", 0.91),
        ("bantuan", "bantuan", "help", 0.93),
    ],
    "German": [
        ("hallo", "hallo", "hello", 0.97),
        ("konto", "konto", "account", 0.95),
        ("guthaben", "guthaben", "balance", 0.92),
        ("zahlung", "zahlung", "payment", 0.91),
        ("problem", "problem", "problem", 0.90),
        ("hilfe", "hilfe", "help", 0.94),
    ],
    "Japanese": [
        ("こんにちは", "konnichiwa", "hello", 0.95),
        ("口座", "kouza", "account", 0.92),
        ("残高", "zandaka", "balance", 0.90),
        ("支払い", "shiharai", "payment", 0.89),
        ("問題", "mondai", "problem", 0.91),
        ("手伝い", "tetsudai", "help", 0.92),
    ],
    "Swahili": [
        ("habari", "habari", "hello", 0.95),
        ("akaunti", "akaunti", "account", 0.92),
        ("baki", "baki", "balance", 0.90),
        ("malipo", "malipo", "payment", 0.89),
        ("tatizo", "tatizo", "problem", 0.90),
        ("msaada", "msaada", "help", 0.93),
    ],
    "Turkish": [
        ("merhaba", "merhaba", "hello", 0.96),
        ("hesap", "hesap", "account", 0.94),
        ("bakiye", "bakiye", "balance", 0.92),
        ("ödeme", "odeme", "payment", 0.90),
        ("sorun", "sorun", "problem", 0.91),
        ("yardım", "yardim", "help", 0.93),
    ],
    "Italian": [
        ("ciao", "ciao", "hello", 0.96),
        ("conto", "conto", "account", 0.94),
        ("saldo", "saldo", "balance", 0.92),
        ("pagamento", "pagamento", "payment", 0.91),
        ("problema", "problema", "problem", 0.90),
        ("aiuto", "aiuto", "help", 0.94),
    ],
    "Korean": [
        ("안녕하세요", "annyeonghaseyo", "hello", 0.95),
        ("계좌", "gyeoja", "account", 0.92),
        ("잔액", "janyak", "balance", 0.90),
        ("결제", "gyeolje", "payment", 0.89),
        ("문제", "munje", "problem", 0.91),
        ("도움", "doum", "help", 0.92),
    ],
    "Vietnamese": [
        ("xin chào", "xin chao", "hello", 0.95),
        ("tài khoản", "tai khoan", "account", 0.92),
        ("số dư", "so du", "balance", 0.90),
        ("thanh toán", "thanh toan", "payment", 0.89),
        ("vấn đề", "van de", "problem", 0.90),
        ("giúp đỡ", "giup do", "help", 0.92),
    ],
    "Punjabi": [
        ("ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "sat sri akal", "hello", 0.94),
        ("ਖਾਤਾ", "khata", "account", 0.91),
        ("ਬੈਲਂਸ", "balance", "balance", 0.89),
        ("ਭੁਗਤਾਨ", "bhugtan", "payment", 0.88),
        ("ਸਮੱਸਿਆ", "samasya", "problem", 0.90),
        ("ਮਦਦ", "madad", "help", 0.92),
    ],
    "Marathi": [
        ("नमस्कार", "namaskar", "hello", 0.95),
        ("खाते", "khate", "account", 0.92),
        ("शिल्लक", "shillak", "balance", 0.90),
        ("देय", "deya", "payment", 0.89),
        ("अडचण", "adchan", "problem", 0.90),
        ("मदत", "madat", "help", 0.92),
    ],
    "Telugu": [
        ("నమస్కారం", "namaskaram", "hello", 0.94),
        ("ఖాతా", "khata", "account", 0.91),
        ("బ్యాలెన్స్", "balance", "balance", 0.89),
        ("చెల్లింపు", "chellimpu", "payment", 0.88),
        ("సమస్య", "samasya", "problem", 0.90),
        ("సహాయం", "sahayam", "help", 0.92),
    ],
    "Tamil": [
        ("வணக்கம்", "vanakkam", "hello", 0.94),
        ("கணக்கு", "kanakku", "account", 0.91),
        ("எச்சில்", "echchil", "balance", 0.89),
        ("கட்டணம்", "kattanum", "payment", 0.88),
        ("பிரச்சனை", "pirachanai", "problem", 0.90),
        ("உதவி", "udhavi", "help", 0.92),
    ],
    "Gujarati": [
        ("નમસ્તે", "namaste", "hello", 0.94),
        ("ખાતું", "khatu", "account", 0.91),
        ("બેલેન્સ", "balance", "balance", 0.89),
        ("ચુકવણી", "chukvani", "payment", 0.88),
        ("સમસ્યા", "samasya", "problem", 0.90),
        ("મદદ", "madad", "help", 0.92),
    ],
    "Kannada": [
        ("ನಮಸ್ಕಾರ", "namaskara", "hello", 0.94),
        ("ಖಾತೆ", "khate", "account", 0.91),
        ("ಬ್ಯಾಲೆನ್ಸ್", "balance", "balance", 0.89),
        ("ಪಾವತಿ", "pavati", "payment", 0.88),
        ("ಸಮಸ್ಯೆ", "samasye", "problem", 0.90),
        ("ಸಹಾಯ", "sahaya", "help", 0.92),
    ],
    "Malayalam": [
        ("നമസ്കാരം", "namaskaram", "hello", 0.94),
        ("അക്കൗണ്ട്", "account", "account", 0.91),
        ("ബാലൻസ്", "balance", "balance", 0.89),
        ("പേയ്മെന്റ്", "payment", "payment", 0.88),
        ("പ്രശ്നം", "prashnam", "problem", 0.90),
        ("സഹായം", "sahayam", "help", 0.92),
    ],
    "Odia": [
        ("ନମସ୍କାର", "namaskara", "hello", 0.93),
        ("ଖାତା", "khata", "account", 0.90),
        ("ବ୍ୟାଲେନ୍ସ", "balance", "balance", 0.88),
        ("ଦେୟ", "deya", "payment", 0.87),
        ("ସମସ୍ୟା", "samasya", "problem", 0.89),
        ("ସାହାଯ୍ୟ", "sahajya", "help", 0.91),
    ],
    "Persian": [
        ("سلام", "salam", "hello", 0.95),
        ("حساب", "hesab", "account", 0.92),
        ("موجودی", "mowjoodi", "balance", 0.90),
        ("پرداخت", "pardakht", "payment", 0.89),
        ("مشکل", "moshkel", "problem", 0.90),
        ("کمک", "komak", "help", 0.92),
    ],
    "Thai": [
        ("สวัสดี", "sawasdee", "hello", 0.94),
        ("บัญชี", "banchi", "account", 0.91),
        ("ยอดคงเหลือ", "yod khong luea", "balance", 0.89),
        ("การชำระเงิน", "kan cham ra ngan", "payment", 0.88),
        ("ปัญหา", "panha", "problem", 0.90),
        ("ช่วยเหลือ", "chuay luea", "help", 0.91),
    ],
    "Filipino": [
        ("kamusta", "kamusta", "hello", 0.95),
        ("account", "account", "account", 0.93),
        ("balanse", "balanse", "balance", 0.90),
        ("bayad", "bayad", "payment", 0.89),
        ("problema", "problema", "problem", 0.90),
        ("tulong", "tulong", "help", 0.92),
    ],
    "Polish": [
        ("cześć", "czesc", "hello", 0.95),
        ("konto", "konto", "account", 0.93),
        ("saldo", "saldo", "balance", 0.91),
        ("płatność", "platnosc", "payment", 0.90),
        ("problem", "problem", "problem", 0.89),
        ("pomoc", "pomoc", "help", 0.92),
    ],
}


# ======================================================================
# DEMO CALL SPECIFICATIONS
# ======================================================================

CALL_SPECS: List[Dict[str, Any]] = [
    {
        "call_id": "GL-001",
        "primary": "Urdu",
        "secondary": ["English", "Roman Urdu"],
        "consent_ref": "CONSENT-9F3C",
        "notice_given": True,
        "speaker_roles": {"speaker_1": "agent", "speaker_2": "caller"},
        "region": "Punjab",
        "accent": "Lahore urban",
        "age_band": "30-45",
        "caller_gender": "male",
        "phone_quality": "good",
        "background_noise": "low",
        "turns": [("speaker_1", 7), ("speaker_2", 10), ("speaker_1", 8)],
    },
    {
        "call_id": "GL-002",
        "primary": "Hindi",
        "secondary": ["English"],
        "consent_ref": "CONSENT-2B81",
        "notice_given": True,
        "speaker_roles": None,
        "region": "North India",
        "accent": "Delhi urban",
        "age_band": "46-60",
        "caller_gender": "female",
        "phone_quality": "medium",
        "background_noise": "medium",
        "turns": [("speaker_1", 8), ("speaker_2", 9)],
    },
    {
        "call_id": "GL-003",
        "primary": "Mandarin Chinese",
        "secondary": ["English"],
        "consent_ref": "CONSENT-77AA",
        "notice_given": True,
        "speaker_roles": {"speaker_1": "agent", "speaker_2": "caller"},
        "region": "China",
        "accent": "Beijing standard",
        "age_band": "18-29",
        "caller_gender": "male",
        "phone_quality": "good",
        "background_noise": "low",
        "turns": [("speaker_1", 7), ("speaker_2", 8), ("speaker_1", 6)],
    },
    {
        "call_id": "GL-004",
        "primary": "Arabic",
        "secondary": ["French", "English"],
        "consent_ref": "CONSENT-5E12",
        "notice_given": True,
        "speaker_roles": {"speaker_1": "agent", "speaker_2": "caller"},
        "region": "MENA",
        "accent": "Casablanca mixed",
        "age_band": "30-45",
        "caller_gender": "female",
        "phone_quality": "medium",
        "background_noise": "medium",
        "turns": [("speaker_1", 8), ("speaker_2", 10)],
    },
    {
        "call_id": "GL-005",
        "primary": "Spanish",
        "secondary": ["Portuguese", "English"],
        "consent_ref": "CONSENT-C3D4",
        "notice_given": True,
        "speaker_roles": None,
        "region": "Latin America",
        "accent": "Bogota urban",
        "age_band": "46-60",
        "caller_gender": "male",
        "phone_quality": "good",
        "background_noise": "low",
        "turns": [("speaker_1", 7), ("speaker_2", 9)],
    },
    {
        "call_id": "GL-006",
        "primary": "Bengali",
        "secondary": ["English"],
        "consent_ref": "",
        "notice_given": True,
        "speaker_roles": {"speaker_1": "agent", "speaker_2": "caller"},
        "region": "Bangladesh",
        "accent": "Dhaka urban",
        "age_band": "18-29",
        "caller_gender": "female",
        "phone_quality": "poor",
        "background_noise": "high",
        "turns": [("speaker_1", 6), ("speaker_2", 8)],
    },
    {
        "call_id": "GL-007",
        "primary": "Russian",
        "secondary": ["English"],
        "consent_ref": "CONSENT-RU88",
        "notice_given": True,
        "speaker_roles": {"speaker_1": "agent", "speaker_2": "caller"},
        "region": "Eastern Europe",
        "accent": "Moscow standard",
        "age_band": "30-45",
        "caller_gender": "male",
        "phone_quality": "medium",
        "background_noise": "medium",
        "turns": [("speaker_1", 7), ("speaker_2", 8)],
    },
    {
        "call_id": "GL-008",
        "primary": "Japanese",
        "secondary": ["English"],
        "consent_ref": "CONSENT-JP45",
        "notice_given": False,
        "speaker_roles": {"speaker_1": "agent", "speaker_2": "caller"},
        "region": "Japan",
        "accent": "Tokyo standard",
        "age_band": "46-60",
        "caller_gender": "female",
        "phone_quality": "good",
        "background_noise": "low",
        "turns": [("speaker_1", 6), ("speaker_2", 7)],
    },
    {
        "call_id": "GL-009",
        "primary": "Swahili",
        "secondary": ["French", "English"],
        "consent_ref": "CONSENT-SW19",
        "notice_given": True,
        "speaker_roles": None,
        "region": "East Africa",
        "accent": "Nairobi urban",
        "age_band": "18-29",
        "caller_gender": "male",
        "phone_quality": "medium",
        "background_noise": "high",
        "turns": [("speaker_1", 8), ("speaker_2", 9)],
    },
    {
        "call_id": "GL-010",
        "primary": "Turkish",
        "secondary": ["German", "English"],
        "consent_ref": "CONSENT-TR73",
        "notice_given": True,
        "speaker_roles": {"speaker_1": "agent", "speaker_2": "caller"},
        "region": "West Asia / Europe",
        "accent": "Istanbul urban",
        "age_band": "30-45",
        "caller_gender": "female",
        "phone_quality": "good",
        "background_noise": "low",
        "turns": [("speaker_1", 7), ("speaker_2", 8)],
    },
    {
        "call_id": "GL-011",
        "primary": "Korean",
        "secondary": ["English"],
        "consent_ref": "CONSENT-KR22",
        "notice_given": True,
        "speaker_roles": {"speaker_1": "agent", "speaker_2": "caller"},
        "region": "East Asia",
        "accent": "Seoul standard",
        "age_band": "18-29",
        "caller_gender": "male",
        "phone_quality": "good",
        "background_noise": "low",
        "turns": [("speaker_1", 6), ("speaker_2", 8)],
    },
    {
        "call_id": "GL-012",
        "primary": "Tamil",
        "secondary": ["Telugu", "Hindi", "English"],
        "consent_ref": "CONSENT-TA55",
        "notice_given": True,
        "speaker_roles": None,
        "region": "South India",
        "accent": "Chennai urban",
        "age_band": "46-60",
        "caller_gender": "female",
        "phone_quality": "medium",
        "background_noise": "medium",
        "turns": [("speaker_1", 8), ("speaker_2", 10)],
    },
    {
        "call_id": "GL-013",
        "primary": "Persian",
        "secondary": ["Arabic", "English"],
        "consent_ref": "CONSENT-FA31",
        "notice_given": True,
        "speaker_roles": {"speaker_1": "agent", "speaker_2": "caller"},
        "region": "West Asia",
        "accent": "Tehran urban",
        "age_band": "30-45",
        "caller_gender": "male",
        "phone_quality": "medium",
        "background_noise": "medium",
        "turns": [("speaker_1", 7), ("speaker_2", 8)],
    },
    {
        "call_id": "GL-014",
        "primary": "Thai",
        "secondary": ["English"],
        "consent_ref": "CONSENT-TH64",
        "notice_given": True,
        "speaker_roles": None,
        "region": "Southeast Asia",
        "accent": "Bangkok urban",
        "age_band": "18-29",
        "caller_gender": "female",
        "phone_quality": "good",
        "background_noise": "low",
        "turns": [("speaker_1", 6), ("speaker_2", 7)],
    },
    {
        "call_id": "GL-015",
        "primary": "Filipino",
        "secondary": ["English", "Spanish"],
        "consent_ref": "CONSENT-PH90",
        "notice_given": True,
        "speaker_roles": {"speaker_1": "agent", "speaker_2": "caller"},
        "region": "Southeast Asia",
        "accent": "Manila urban",
        "age_band": "30-45",
        "caller_gender": "male",
        "phone_quality": "medium",
        "background_noise": "medium",
        "turns": [("speaker_1", 7), ("speaker_2", 9)],
    },
    {
        "call_id": "GL-016",
        "primary": "Polish",
        "secondary": ["German", "English"],
        "consent_ref": "CONSENT-PL18",
        "notice_given": True,
        "speaker_roles": None,
        "region": "Europe",
        "accent": "Warsaw urban",
        "age_band": "46-60",
        "caller_gender": "female",
        "phone_quality": "good",
        "background_noise": "low",
        "turns": [("speaker_1", 6), ("speaker_2", 8)],
    },
]


# ======================================================================
# SMALL UTILITIES
# ======================================================================

def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def esc(s: Any) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", str(text))

    out_chars = []
    for ch in text:
        if unicodedata.category(ch) == "Nd":
            try:
                out_chars.append(str(unicodedata.digit(ch)))
            except Exception:
                out_chars.append(ch)
        else:
            out_chars.append(ch)
    text = "".join(out_chars)

    for src, dst in PUNCT_MAP.items():
        text = text.replace(src, dst)

    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return text


def disallowed_chars(text: str) -> List[str]:
    allowed_punct = set(".,!?;:\"'()[]{}-/\\%&*+=<>@#$_~`،؛؟۔")
    bad = []
    for ch in str(text):
        cat = unicodedata.category(ch)
        if cat.startswith("L") or cat.startswith("N") or cat == "Zs":
            continue
        if cat.startswith("P") and ch in allowed_punct:
            continue
        if ch in "\n\t":
            continue
        bad.append(ch)
    return sorted(set(bad))


def language_meta(name: str) -> Dict[str, Any]:
    return LANG_BY_NAME.get(
        name,
        {
            "name": name,
            "iso": "und",
            "scripts": ["Latin"],
            "region": "Unknown",
            "speakers_millions": 1,
            "rtl": False,
            "low_resource": True,
        },
    )


def generic_role(speaker_label: str, roles: Optional[Dict[str, str]] = None) -> str:
    if roles and speaker_label in roles:
        return roles[speaker_label]
    try:
        num = str(int(speaker_label.split("_")[-1]))
    except Exception:
        num = "1"
    return f"Speaker {num}"


def has_valid_consent(call: Dict[str, Any]) -> bool:
    ref = str(call.get("consent_ref", "") or "").strip()
    return bool(ref) and bool(call.get("notice_given", True))


def ensure_consent(call: Dict[str, Any], require: bool = True) -> bool:
    if not require:
        return True
    if not has_valid_consent(call):
        if not str(call.get("consent_ref", "") or "").strip():
            raise ConsentError("Transcription refused: missing recorded consent reference.")
        raise ConsentError("Transcription refused: recorded parties were not properly notified.")
    return True


# ======================================================================
# AUDIO HELPERS
# ======================================================================

def float_to_wav(audio: np.ndarray, sr: int = 16000) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    pcm = (audio * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def decode_wav_bytes(data: bytes) -> Tuple[np.ndarray, int]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if sw == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 1:
        audio = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    else:
        raise ValueError("Only 8-bit or 16-bit WAV is supported in this demo.")

    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1)

    return audio.astype(np.float32), sr


def resample_to_16k_mono(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr == 16000:
        return audio.astype(np.float32)
    if len(audio) == 0:
        return np.zeros(16000, dtype=np.float32)
    duration = len(audio) / float(sr)
    n = max(1, int(round(duration * 16000)))
    x_old = np.linspace(0.0, 1.0, len(audio))
    x_new = np.linspace(0.0, 1.0, n)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def frame_rms_array(
    audio: np.ndarray,
    sr: int,
    frame_s: float = 0.02,
    hop_s: float = 0.01,
) -> np.ndarray:
    if len(audio) == 0:
        return np.array([0.0], dtype=np.float32)
    frame = max(1, int(frame_s * sr))
    hop = max(1, int(hop_s * sr))
    vals = []
    for i in range(0, max(1, len(audio) - frame + 1), hop):
        chunk = audio[i : i + frame]
        vals.append(float(np.sqrt(np.mean(chunk * chunk))) if len(chunk) else 0.0)
    return np.array(vals, dtype=np.float32)


def estimate_snr(audio: np.ndarray, sr: int = 16000) -> Tuple[float, float]:
    rms = frame_rms_array(audio, sr)
    noise = float(np.percentile(rms, 10)) if len(rms) else 0.0
    speech = float(np.percentile(rms, 90)) if len(rms) else 0.0
    if noise <= 1e-6:
        snr = 100.0
    else:
        snr = 20.0 * math.log10(max(speech, 1e-6) / noise)
    return float(snr), float(noise)


def light_noise_reduce(audio: np.ndarray, noise_level: float) -> np.ndarray:
    processed = np.sign(audio) * np.maximum(np.abs(audio) - noise_level * 0.35, 0.0)
    return processed.astype(np.float32)


def vad_segments_from_array(
    audio: np.ndarray,
    sr: int = 16000,
    max_segment_s: Optional[float] = None,
    silence_threshold: Optional[float] = None,
    min_speech_s: Optional[float] = None,
    pad_s: Optional[float] = None,
    merge_gap_s: Optional[float] = None,
) -> List[Tuple[float, float]]:
    cfg = CONFIG["vad"]
    max_segment_s = cfg["max_segment_s"] if max_segment_s is None else max_segment_s
    silence_threshold = cfg["silence_threshold"] if silence_threshold is None else silence_threshold
    min_speech_s = cfg["min_speech_s"] if min_speech_s is None else min_speech_s
    pad_s = cfg["pad_s"] if pad_s is None else pad_s
    merge_gap_s = cfg["merge_gap_s"] if merge_gap_s is None else merge_gap_s

    if len(audio) == 0:
        return []

    hop_s = 0.01
    rms = frame_rms_array(audio, sr, frame_s=0.02, hop_s=hop_s)
    speech = rms > silence_threshold

    groups = []
    start = None
    last_true = None
    for i, s in enumerate(speech):
        if s:
            if start is None:
                start = i
            last_true = i
        else:
            if start is not None and (i - last_true) * hop_s > merge_gap_s:
                groups.append((start, last_true))
                start = None
    if start is not None:
        groups.append((start, last_true))

    duration = len(audio) / float(sr)
    intervals = []
    for g_start, g_end in groups:
        stime = g_start * hop_s
        etime = (g_end + 1) * hop_s
        if etime - stime < min_speech_s:
            continue
        stime = max(0.0, stime - pad_s)
        etime = min(duration, etime + pad_s)

        cur = stime
        while cur < etime:
            seg_end = min(etime, cur + max_segment_s)
            intervals.append((round(cur, 3), round(seg_end, 3)))
            cur = seg_end

    return intervals


def prepare_audio_from_bytes(wav_bytes: bytes) -> Dict[str, Any]:
    audio, sr = decode_wav_bytes(wav_bytes)
    audio = resample_to_16k_mono(audio, sr)
    snr, noise_level = estimate_snr(audio, 16000)
    reduced = False
    if snr < CONFIG["vad"]["snr_threshold_db"]:
        audio = light_noise_reduce(audio, noise_level)
        reduced = True
    segments = vad_segments_from_array(audio, 16000)
    return {
        "wav": float_to_wav(audio, 16000),
        "segments": segments,
        "snr_db": round(snr, 2),
        "noise_reduced": reduced,
    }


def synth_test_audio_array(sr: int = 16000) -> np.ndarray:
    duration = 10.0
    n = int(sr * duration)
    audio = np.zeros(n, dtype=np.float32)
    bursts = [(1.0, 2.5), (5.5, 7.0)]
    for s, e in bursts:
        i0 = int(s * sr)
        i1 = int(e * sr)
        t = np.arange(i1 - i0) / float(sr)
        tone = np.sin(2 * np.pi * 220 * t) + 0.30 * np.sin(2 * np.pi * 440 * t)
        env = np.ones_like(t)
        ramp = int(0.01 * sr)
        if ramp > 0 and len(env) > 2 * ramp:
            env[:ramp] = np.linspace(0, 1, ramp)
            env[-ramp:] = np.linspace(1, 0, ramp)
        audio[i0:i1] = 0.25 * tone * env
    return audio


def segment_audio_bytes(seg: Dict[str, Any]) -> bytes:
    sr = 16000
    duration = max(1.0, float(seg["end"] - seg["start"]) + 0.7)
    n = int(sr * duration)
    audio = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(int(seg.get("audio_seed", 1)))
    base_freq = 170 + (int(seg.get("audio_seed", 1)) % 90)
    offset = 0.25

    for w in seg.get("words", []):
        ws = float(w["start"]) - float(seg["start"]) + offset
        we = float(w["end"]) - float(seg["start"]) + offset
        i0 = max(0, int(ws * sr))
        i1 = min(n, int(we * sr))
        if i1 <= i0:
            continue
        t = np.arange(i1 - i0) / float(sr)
        env = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 0.7
        tone = (
            np.sin(2 * np.pi * base_freq * t)
            + 0.35 * np.sin(2 * np.pi * base_freq * 2 * t)
            + 0.15 * np.sin(2 * np.pi * base_freq * 3 * t)
        )
        amp = 0.10 + 0.20 * float(w.get("conf", 0.8))
        audio[i0:i1] += amp * env * tone

    audio += rng.normal(0.0, 0.0015, n).astype(np.float32)
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak > 0:
        audio = audio * (0.82 / peak)
    return float_to_wav(audio, sr)


# ======================================================================
# TRANSCRIPT MODEL HELPERS
# ======================================================================

def make_words_from_mix(
    start: float,
    primary: str,
    secondary: List[str],
    rng: np.random.Generator,
    n_words: int,
) -> Tuple[List[Dict[str, Any]], float]:
    langs = [primary] + [x for x in secondary if x != primary]
    if len(langs) == 1:
        weights = [1.0]
    else:
        weights = [0.55] + [0.45 / max(1, len(langs) - 1) for _ in range(len(langs) - 1)]

    weights = np.array(weights, dtype=float)
    weights /= weights.sum()

    t = float(start)
    words = []
    for _ in range(max(4, int(n_words))):
        lang = str(rng.choice(langs, p=weights))
        lib = TOKEN_LIBRARY.get(lang, TOKEN_LIBRARY["English"])
        native, roman, english, base_conf = lib[int(rng.integers(0, len(lib)))]

        conf = float(np.clip(base_conf + rng.normal(0.0, 0.025), 0.35, 0.995))
        if rng.random() < 0.12:
            conf = float(rng.uniform(0.42, 0.72))

        display_len = len(roman or native or english)
        dur = float(np.clip(0.14 + 0.035 * display_len + rng.normal(0.0, 0.03), 0.12, 0.75))

        words.append(
            {
                "native": native,
                "roman": roman,
                "english": english,
                "lang": lang,
                "start": round(t, 3),
                "end": round(t + dur, 3),
                "conf": round(conf, 4),
            }
        )
        t += dur + float(np.clip(rng.normal(0.05, 0.02), 0.01, 0.18))

    return words, t


def make_segment(
    segment_id: str,
    call_id: str,
    speaker_label: str,
    start: float,
    primary: str,
    secondary: List[str],
    rng: np.random.Generator,
    n_words: int,
    seed: int,
) -> Dict[str, Any]:
    words, end = make_words_from_mix(start, primary, secondary, rng, n_words)
    confs = [float(w["conf"]) for w in words] if words else [0.0]
    avg_conf = float(np.mean(confs))

    counts: Dict[str, int] = {}
    families: Dict[str, int] = {}
    for w in words:
        lang = w["lang"]
        fam = "Urdu" if lang in URDU_FAMILY else lang
        counts[lang] = counts.get(lang, 0) + 1
        families[fam] = families.get(fam, 0) + 1

    minority = min(families.values()) if families else 0
    has_english = "English" in families
    has_non_english = any(f != "English" for f in families)
    code_switched = (
        len(families) > 1
        and (
            minority / max(1, len(words)) >= 0.18
            or (has_english and has_non_english)
        )
    )

    if code_switched:
        language = "Code-switched"
    elif families:
        language = max(families, key=lambda k: families[k])
    else:
        language = primary

    return {
        "segment_id": segment_id,
        "call_id": call_id,
        "speaker_label": speaker_label,
        "speaker_code": f"{call_id}_{speaker_label}",
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "words": words,
        "confidence": round(avg_conf, 4),
        "language": language,
        "languages_present": sorted(counts.keys()),
        "code_switched": bool(code_switched),
        "second_transcriber": False,
        "disagreements": [],
        "audio_seed": int(seed),
    }


def apply_diarisation(call: Dict[str, Any]) -> None:
    turns = []
    for seg in call.get("segments", []):
        turns.append(
            {
                "speaker": seg["speaker_label"],
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "role": generic_role(seg["speaker_label"], call.get("speaker_roles")),
            }
        )
        seg["speaker_role"] = generic_role(seg["speaker_label"], call.get("speaker_roles"))

    call["speaker_turns"] = turns

    for seg in call.get("segments", []):
        for w in seg.get("words", []):
            mid = (float(w["start"]) + float(w["end"])) / 2.0
            turn = next((t for t in turns if t["start"] <= mid <= t["end"]), None)
            if turn is None:
                turn = {
                    "speaker": seg["speaker_label"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "role": seg.get("speaker_role", generic_role(seg["speaker_label"], call.get("speaker_roles"))),
                }
            w["speaker"] = turn["speaker"]
            w["role"] = turn["role"]


def build_demo_calls() -> List[Dict[str, Any]]:
    calls = []
    global_seg_index = 0
    second_every = max(1, int(round(1.0 / max(0.01, CONFIG["evaluation"]["second_transcriber_rate"]))))

    for ci, spec in enumerate(CALL_SPECS):
        rng = np.random.default_rng(2026 + ci * 101)
        call = {
            "call_id": spec["call_id"],
            "primary_language": spec["primary"],
            "secondary_languages": spec["secondary"],
            "consent_ref": spec["consent_ref"],
            "notice_given": spec["notice_given"],
            "speaker_roles": spec["speaker_roles"],
            "region": spec["region"],
            "accent": spec["accent"],
            "age_band": spec["age_band"],
            "caller_gender": spec["caller_gender"],
            "phone_quality": spec["phone_quality"],
            "background_noise": spec["background_noise"],
            "segments": [],
        }

        seg_start = 0.25
        for si, (speaker_label, n_words) in enumerate(spec["turns"]):
            seg_id = f"{call['call_id']}-S{si+1:02d}"
            seed = 1000 + ci * 37 + si * 11
            seg = make_segment(
                segment_id=seg_id,
                call_id=call["call_id"],
                speaker_label=speaker_label,
                start=seg_start,
                primary=spec["primary"],
                secondary=spec["secondary"],
                rng=rng,
                n_words=n_words,
                seed=seed,
            )

            if global_seg_index % second_every == 0:
                seg["second_transcriber"] = True
                if seg["words"]:
                    dis_idx = int(np.argmin([w["conf"] for w in seg["words"]]))
                    original = seg["words"][dis_idx]["roman"] or seg["words"][dis_idx]["native"]
                    seg["disagreements"].append(
                        {
                            "word_index": dis_idx,
                            "annotator_a": original,
                            "annotator_b": f"{original} (variant)",
                            "resolved": original,
                            "note": "Second transcriber disagreement recorded.",
                        }
                    )

            call["segments"].append(seg)
            seg_start = float(seg["end"]) + 0.35
            global_seg_index += 1

        apply_diarisation(call)
        calls.append(call)

    return calls


# ======================================================================
# VALIDATION, SPLITS, TRANSCRIPT OUTPUT, REPORTING
# ======================================================================

def validate_segment(seg: Dict[str, Any]) -> List[str]:
    errors = []
    words = seg.get("words", [])
    if not words:
        errors.append("empty segment")
        return errors
    for i, w in enumerate(words):
        text = normalize_text(w.get("roman", "") or w.get("native", "") or w.get("english", ""))
        if not text:
            errors.append(f"empty word at index {i}")
    return errors


def validate_split(train: List[Dict[str, Any]], test: List[Dict[str, Any]]) -> List[str]:
    errors = []
    train_speakers = {x.get("speaker_code") for x in train if x.get("speaker_code")}
    test_speakers = {x.get("speaker_code") for x in test if x.get("speaker_code")}
    speaker_overlap = train_speakers & test_speakers
    if speaker_overlap:
        errors.append(f"speaker leakage: {sorted(speaker_overlap)}")

    train_calls = {x.get("call_id") for x in train if x.get("call_id")}
    test_calls = {x.get("call_id") for x in test if x.get("call_id")}
    call_overlap = train_calls & test_calls
    if call_overlap:
        errors.append(f"call leakage: {sorted(call_overlap)}")

    return errors


def build_splits(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_segs = []
    for c in calls:
        for s in c.get("segments", []):
            all_segs.append(
                {
                    "speaker_code": s["speaker_code"],
                    "call_id": s["call_id"],
                    "segment_id": s["segment_id"],
                }
            )

    call_ids = sorted({x["call_id"] for x in all_segs})
    train_calls = set(call_ids[::2])
    test_calls = set(call_ids[1::2])

    good_train = [x for x in all_segs if x["call_id"] in train_calls]
    good_test = [x for x in all_segs if x["call_id"] in test_calls]

    bad_train = [{"speaker_code": "S1", "call_id": "C1", "segment_id": "X1"}]
    bad_test = [{"speaker_code": "S1", "call_id": "C2", "segment_id": "X2"}]

    return {
        "good_train": good_train,
        "good_test": good_test,
        "good_errors": validate_split(good_train, good_test),
        "bad_train": bad_train,
        "bad_test": bad_test,
        "bad_errors": validate_split(bad_train, bad_test),
        "train_calls": sorted(train_calls),
        "test_calls": sorted(test_calls),
    }


def word_display(w: Dict[str, Any], script_mode: str) -> str:
    if script_mode == "english":
        text = w.get("english") or w.get("roman") or w.get("native")
    elif script_mode == "roman":
        text = w.get("roman") or w.get("english") or w.get("native")
    else:
        text = w.get("native") or w.get("roman") or w.get("english")
    return normalize_text(text or "")


def make_transcript_json(call: Dict[str, Any], script_mode: str) -> Dict[str, Any]:
    words = []
    for seg in sorted(call.get("segments", []), key=lambda s: s["start"]):
        for w in seg.get("words", []):
            out = {
                "text": word_display(w, script_mode),
                "start": float(w.get("start", 0.0)),
                "end": float(w.get("end", 0.0)),
                "confidence": float(w.get("conf", 0.0)),
                "language": w.get("lang", "Unknown"),
                "speaker": w.get("role") or generic_role(w.get("speaker", "speaker_1")),
                "speaker_label": w.get("speaker", seg.get("speaker_label", "speaker_1")),
                "segment_id": seg.get("segment_id", ""),
                "low_confidence": float(w.get("conf", 0.0)) < float(CONFIG["confidence"]["word_low_threshold"]),
            }
            words.append(out)

    return {
        "call_id": call["call_id"],
        "consent_ref": call.get("consent_ref", ""),
        "notice_given": bool(call.get("notice_given", True)),
        "script_mode": script_mode,
        "primary_language": call.get("primary_language", "Unknown"),
        "secondary_languages": call.get("secondary_languages", []),
        "generated_at": now_iso(),
        "words": words,
    }


def format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"[{m:02d}:{s:02d}]"


def readable_transcript(call: Dict[str, Any], script_mode: str) -> str:
    lines = []
    next_marker = 0.0
    for seg in sorted(call.get("segments", []), key=lambda s: s["start"]):
        while float(seg["start"]) >= next_marker:
            lines.append(format_timestamp(next_marker))
            next_marker += 30.0

        text = " ".join(word_display(w, script_mode) for w in seg.get("words", []))
        role = seg.get("speaker_role", generic_role(seg["speaker_label"], call.get("speaker_roles")))
        lines.append(f"{role} [{float(seg['start']):.1f}s-{float(seg['end']):.1f}s] ({seg.get('language','')}): {text}")
    return "\n".join(lines)


def readable_words_only(call: Dict[str, Any], script_mode: str) -> str:
    parts = []
    for seg in sorted(call.get("segments", []), key=lambda s: s["start"]):
        for w in seg.get("words", []):
            parts.append(word_display(w, script_mode))
    return " ".join(parts)


def call_report(call: Dict[str, Any]) -> Dict[str, Any]:
    words = [w for seg in call.get("segments", []) for w in seg.get("words", [])]
    segments = call.get("segments", [])

    if words:
        avg_conf = float(np.mean([float(w["conf"]) for w in words]))
        low_share = float(
            sum(1 for w in words if float(w["conf"]) < float(CONFIG["confidence"]["word_low_threshold"]))
            / len(words)
        )
    else:
        avg_conf = 0.0
        low_share = 0.0

    code_share = 0.0
    if segments:
        code_share = float(sum(1 for s in segments if s.get("code_switched")) / len(segments))

    flag = (
        avg_conf < float(CONFIG["confidence"]["flag_avg_confidence_below"])
        or low_share > 0.25
    )

    return {
        "call_id": call["call_id"],
        "primary_language": call.get("primary_language", ""),
        "consent_valid": has_valid_consent(call),
        "average_word_confidence": round(avg_conf, 4),
        "low_confidence_word_share": round(low_share, 4),
        "code_switched_segment_share": round(code_share, 4),
        "flag_for_human_review": bool(flag),
        "auto_accept_eligible": bool(
            has_valid_consent(call)
            and avg_conf >= float(CONFIG["confidence"]["auto_accept_min"])
            and not flag
        ),
    }


def run_transcription(
    call: Dict[str, Any],
    script_mode: str,
    require_consent: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    ensure_consent(call, require_consent)

    transcript = make_transcript_json(call, script_mode)
    report = call_report(call)

    prep_info = []
    for seg in call.get("segments", [])[:3]:
        prep = prepare_audio_from_bytes(segment_audio_bytes(seg))
        prep_info.append(
            {
                "segment_id": seg["segment_id"],
                "snr_db": prep["snr_db"],
                "noise_reduced": prep["noise_reduced"],
                "vad_segments": prep["segments"],
                "language": seg.get("language", ""),
                "code_switched": seg.get("code_switched", False),
            }
        )

    return transcript, report, prep_info


def html_transcript_segment(seg: Dict[str, Any], script_mode: str) -> str:
    parts = []
    for w in seg.get("words", []):
        text = word_display(w, script_mode)
        meta = language_meta(w.get("lang", ""))
        rtl = bool(meta.get("rtl", False)) and script_mode != "roman"
        low = float(w.get("conf", 0.0)) < float(CONFIG["confidence"]["word_low_threshold"])
        cls = "word"
        if low:
            cls += " low-conf"
        if rtl:
            cls += " rtl"
        dir_attr = ' dir="rtl"' if rtl else ' dir="ltr"'
        title = f"{w.get('lang','')} · conf {float(w.get('conf',0)):.2f}"
        parts.append(f'<span class="{cls}"{dir_attr} title="{esc(title)}">{esc(text)}</span>')
    return " ".join(parts)


# ======================================================================
# EVALUATION SIMULATION
# ======================================================================

SPEAKER_GROUPS = [
    "Urban standard",
    "Regional accent A",
    "Regional accent B",
    "Rural accent",
    "Elder speaker",
    "Young speaker",
]


def build_eval_dataframe(languages: Optional[List[str]] = None) -> pd.DataFrame:
    langs = [l for l in (languages or LANGUAGE_NAMES) if l in LANG_BY_NAME] or LANGUAGE_NAMES
    rng = np.random.default_rng(2026)
    n = int(CONFIG["evaluation"]["held_out_calls"])

    weights = np.array([max(1, language_meta(l)["speakers_millions"]) for l in langs], dtype=float)
    weights /= weights.sum()
    chosen = rng.choice(langs, size=n, p=weights)

    rows = []
    for i, lang in enumerate(chosen):
        meta = language_meta(lang)
        code_switched = bool(rng.random() < 0.35)
        script = "Latin" if lang == "Roman Urdu" or rng.random() < 0.35 else meta["scripts"][0]
        group = SPEAKER_GROUPS[i % len(SPEAKER_GROUPS)]
        phone = rng.choice(["good", "medium", "poor"], p=[0.45, 0.35, 0.20])
        noise = rng.choice(["low", "medium", "high"], p=[0.45, 0.35, 0.20])

        base = 0.075 if not meta["low_resource"] else 0.125
        if lang == "Roman Urdu":
            base = 0.115
        if code_switched:
            base += 0.065
        if phone == "poor":
            base += 0.030
        elif phone == "medium":
            base += 0.012
        if noise == "high":
            base += 0.022
        elif noise == "medium":
            base += 0.009
        if group in {"Rural accent", "Regional accent A", "Regional accent B"}:
            base += 0.010

        upper = 0.245 if code_switched else 0.220
        wer = float(np.clip(base + rng.normal(0.0, 0.006), 0.035, upper))
        cer = float(np.clip(wer * 0.42 + rng.normal(0.0, 0.004), 0.010, 0.180))

        rows.append(
            {
                "eval_call_id": f"EVAL-{i+1:03d}",
                "language": lang,
                "script": script,
                "code_switched": code_switched,
                "speaker_group": group,
                "phone_quality": phone,
                "background_noise": noise,
                "annotator_a": f"ANOT-A-{i % 7 + 1:02d}",
                "annotator_b": f"ANOT-B-{i % 9 + 1:02d}",
                "third_resolver_used": bool(i % 11 == 0),
                "wer": round(wer, 4),
                "cer": round(cer, 4),
                "inter_annotator_agreement": round(float(0.91 + rng.normal(0.0, 0.015)), 4),
            }
        )

    return pd.DataFrame(rows)


def build_eval_report(df: pd.DataFrame) -> Tuple[str, Dict[str, Any]]:
    if df.empty:
        return "# No evaluation data\n", {}

    overall_wer = float(df["wer"].mean())
    overall_cer = float(df["cer"].mean())
    code_df = df[df["code_switched"]]
    code_wer = float(code_df["wer"].mean()) if not code_df.empty else 0.0

    lang_means = (df.groupby("language")["wer"].mean() * 100).sort_values(ascending=False)
    group_means = (df.groupby("speaker_group")["wer"].mean() * 100).sort_values(ascending=False)
    max_gap_points = float(max(0.0, group_means.max() - overall_wer * 100)) if not group_means.empty else 0.0

    low_resource_langs = [l for l in df["language"].unique() if language_meta(l).get("low_resource", False)]
    low_resource_wer = float(df[df["language"].isin(low_resource_langs)]["wer"].mean()) if low_resource_langs else 0.0

    targets = CONFIG["evaluation"]["targets"]
    summary = {
        "held_out_calls": int(len(df)),
        "overall_wer_pct": round(overall_wer * 100, 2),
        "overall_cer_pct": round(overall_cer * 100, 2),
        "code_switched_wer_pct": round(code_wer * 100, 2),
        "low_resource_wer_pct": round(low_resource_wer * 100, 2),
        "max_group_gap_points": round(max_gap_points, 2),
        "pass_overall": bool(overall_wer <= targets["overall_wer_max"]),
        "pass_code_switched": bool(code_wer <= targets["code_switched_wer_max"]),
        "pass_group_gap": bool(max_gap_points <= targets["group_gap_points_max"]),
        "pass_low_resource": bool(low_resource_wer <= targets["low_resource_wer_max"]),
    }

    md = f"""# Global ASR Evaluation Report

Held-out calls: {summary['held_out_calls']}

Annotation protocol: two native speakers transcribe each held-out item; a third resolver settles disagreements.

## Headline metrics

- Overall WER: **{summary['overall_wer_pct']}%**
- Overall CER: **{summary['overall_cer_pct']}%**
- Code-switched WER: **{summary['code_switched_wer_pct']}%**
- Low-resource language WER: **{summary['low_resource_wer_pct']}%**
- Worst speaker-group gap vs average: **{summary['max_group_gap_points']} points**

## Targets

- Overall WER below 15%: **{'PASS' if summary['pass_overall'] else 'FAIL'}**
- Code-switched WER below 25%: **{'PASS' if summary['pass_code_switched'] else 'FAIL'}**
- No group more than 5 points worse than average: **{'PASS' if summary['pass_group_gap'] else 'FAIL'}**
- Low-resource WER below 20%: **{'PASS' if summary['pass_low_resource'] else 'FAIL'}**

## WER by language

| Language | WER (%) |
|---|---:|
"""
    for lang, val in lang_means.items():
        md += f"| {lang} | {val:.2f} |\n"

    md += """
## WER by speaker group

| Speaker group | WER (%) |
|---|---:|
"""
    for grp, val in group_means.items():
        md += f"| {grp} | {val:.2f} |\n"

    md += """
## Notes

This demo uses simulated held-out evaluation data. Before formal use, replace with real native-speaker annotations and re-run the same group-level checks.
"""

    return md, summary


# ======================================================================
# INTERNAL TESTS
# ======================================================================

def run_internal_tests() -> pd.DataFrame:
    results = []

    def add(name: str, passed: bool, detail: str = "") -> None:
        results.append({"Test": name, "Result": "PASS" if passed else "FAIL", "Detail": detail})

    # Consent gate
    try:
        ensure_consent({"consent_ref": "", "notice_given": True}, True)
        add("Missing consent reference stops transcription", False, "No exception raised")
    except ConsentError as e:
        add("Missing consent reference stops transcription", True, str(e))

    try:
        ensure_consent({"consent_ref": "X", "notice_given": False}, True)
        add("Missing notice stops transcription", False, "No exception raised")
    except ConsentError as e:
        add("Missing notice stops transcription", True, str(e))

    # Empty segment / empty word
    errs = validate_segment({"words": []})
    add("Empty segment is rejected", len(errs) > 0, str(errs))

    errs = validate_segment({"words": [{"native": "", "roman": "", "english": ""}]})
    add("Empty word is rejected", len(errs) > 0, str(errs))

    # Split leakage
    bad_errors = validate_split(
        [{"speaker_code": "S1", "call_id": "C1"}],
        [{"speaker_code": "S1", "call_id": "C2"}],
    )
    add("Same speaker in train and test fails", len(bad_errors) > 0, str(bad_errors))

    splits = build_splits(st.session_state.calls)
    add("Demo split has no speaker/call leakage", len(splits["good_errors"]) == 0, str(splits["good_errors"]))

    # Text normalisation
    bad = disallowed_chars(normalize_text("Hello 😀 world"))
    add("Characters outside allowed set are flagged", len(bad) > 0, ",".join(bad))

    n1 = normalize_text("café")
    n2 = normalize_text("cafe\u0301")
    add("Equivalent Unicode forms normalise equally", n1 == n2, f"{n1} | {n2}")

    n3 = normalize_text("۴۴۷۱")
    add("Non-ASCII digits normalise to ASCII", n3 == "4471", n3)

    # VAD test
    audio = synth_test_audio_array()
    segs = vad_segments_from_array(audio, 16000)
    expected = [(1.0, 2.5), (5.5, 7.0)]
    vad_pass = len(segs) == 2
    if vad_pass:
        for i in range(2):
            if abs(segs[i][0] - expected[i][0]) > 0.1 or abs(segs[i][1] - expected[i][1]) > 0.1:
                vad_pass = False
    add("Two bursts separated by 3s silence produce two segments", vad_pass, f"segments={segs}")

    # Diarisation
    mini_call = {
        "call_id": "TEST-DIA",
        "consent_ref": "X",
        "notice_given": True,
        "speaker_roles": None,
        "segments": [
            {
                "segment_id": "s1",
                "call_id": "TEST-DIA",
                "speaker_label": "speaker_1",
                "speaker_code": "TEST-DIA_speaker_1",
                "start": 0.0,
                "end": 2.0,
                "confidence": 0.9,
                "code_switched": False,
                "words": [
                    {
                        "native": "test",
                        "roman": "test",
                        "english": "test",
                        "start": 0.5,
                        "end": 0.8,
                        "conf": 0.9,
                        "lang": "English",
                    }
                ],
            }
        ],
    }
    apply_diarisation(mini_call)
    role = mini_call["segments"][0]["words"][0].get("role")
    add("Word inside speaker turn gets generic label", role == "Speaker 1", str(role))

    mini_call2 = copy.deepcopy(mini_call)
    mini_call2["speaker_roles"] = {"speaker_1": "agent"}
    apply_diarisation(mini_call2)
    role2 = mini_call2["segments"][0]["words"][0].get("role")
    add("Metadata maps speaker to role", role2 == "agent", str(role2))

    # Low-confidence flag
    low_call = {
        "call_id": "LOW-CONF",
        "consent_ref": "X",
        "notice_given": True,
        "segments": [
            {
                "segment_id": "s1",
                "code_switched": False,
                "words": [
                    {
                        "conf": 0.50,
                        "lang": "English",
                        "native": "a",
                        "roman": "a",
                        "english": "a",
                        "start": 0.0,
                        "end": 1.0,
                    }
                ],
            }
        ],
    }
    rep = call_report(low_call)
    add("Recording with many low-confidence words is flagged", bool(rep["flag_for_human_review"]), json.dumps(rep))

    # JSON/readable reproduction
    call0 = st.session_state.calls[0]
    js = make_transcript_json(call0, "roman")
    txt_json = " ".join(w["text"] for w in js["words"])
    txt_readable = readable_words_only(call0, "roman")
    add("JSON output reproduces readable transcript word text", txt_json == txt_readable, "")

    # Global language coverage
    required = [
        "English",
        "Mandarin Chinese",
        "Hindi",
        "Spanish",
        "French",
        "Arabic",
        "Bengali",
        "Portuguese",
        "Russian",
        "Urdu",
        "Roman Urdu",
    ]
    missing = [x for x in required if x not in LANG_BY_NAME]
    add("Required global languages are present", len(missing) == 0, ",".join(missing))

    # Code-switch detection
    rng = np.random.default_rng(99)
    cs_seg = make_segment("cs1", "C", "speaker_1", 0.0, "Urdu", ["English"], rng, 10, 1)
    add("Urdu/English mixed segment is marked code-switched", bool(cs_seg["code_switched"]), str(cs_seg["languages_present"]))

    ur_only = make_segment("ur1", "C", "speaker_1", 0.0, "Urdu", ["Roman Urdu"], np.random.default_rng(98), 8, 2)
    add("Urdu/Roman Urdu script mix is not falsely code-switched", not bool(ur_only["code_switched"]), str(ur_only["languages_present"]))

    # Script modes
    ok_modes = True
    for mode in ["native", "roman", "english"]:
        txt = readable_words_only(call0, mode)
        if not txt.strip():
            ok_modes = False
    add("All output script modes produce non-empty text", ok_modes, "")

    # Retention
    d = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=int(CONFIG["retention"]["audio_days"]))
    add("Retention date can be calculated", isinstance(d, dt.datetime), d.date().isoformat())

    return pd.DataFrame(results)


# ======================================================================
# DOCUMENTATION TEXT
# ======================================================================

SCOPE_MD = """# Scope

## Purpose

This system produces transcripts for quality review and record keeping of calls and interviews.

It supports human supervision, audit, search, service improvement, and multilingual analysis.

## Decision boundary

The system does not make decisions about callers.

No eligibility, complaint outcome, account action, legal conclusion, or benefit decision may be based only on automatic transcript text.

## Language mix

The system is designed for global multilingual and code-switched speech, including:

- English
- Mandarin Chinese
- Hindi
- Spanish
- French
- Arabic
- Bengali
- Portuguese
- Russian
- Urdu
- Roman Urdu
- Indonesian
- German
- Japanese
- Swahili
- Turkish
- Italian
- Korean
- Vietnamese
- Punjabi
- Marathi
- Telugu
- Tamil
- Gujarati
- Kannada
- Malayalam
- Odia
- Persian
- Thai
- Filipino
- Polish

## Expected variation

The system should expect:

- male and female speakers
- different age bands
- regional accents and dialects
- urban and rural speech patterns
- phone-line audio quality
- background noise
- mixed-script output needs: native script, Roman/Latin transliteration, and English gloss
- code-switching within a sentence, especially where English mixes with local languages
"""

CONSENT_LAW_MD = """# Consent and Law

## Notice and consent

Every recorded party must be told that the call is recorded and transcribed.

Acceptable evidence is one of:

- a recorded consent statement from each party, or
- a clear notice that the call is recorded and that continuing the call indicates consent, where local law allows that model.

The transcript pipeline must store a consent reference for each recording.

## Local and international rules

Local rules on call recording, notice, and storage of personal information may differ by country, sector, and language community.

A legal review is required before deployment.

The legal review must confirm:

- lawful basis for recording
- notice wording in each supported language
- consent evidence retention
- personal-data handling
- cross-border storage, if any
- subject-access and deletion rights
- labour/union rules where agents are recorded
- sector-specific rules for health, finance, legal, or government calls

## Retention

Default demo retention:

- Audio: 90 days
- Transcripts and confidence reports: 365 days

## Deletion process

Deletion must include:

1. Mark the record as deleted in the operational index.
2. Remove audio and transcript files from primary storage.
3. Request deletion from backup systems according to backup expiry policy.
4. Keep a minimal audit tombstone without personal content where legally required.
5. Record the deletion timestamp and operator/reference.
"""


def generate_languages_yml() -> str:
    lines = [
        "default_locale: en-PK",
        "",
        "supported_languages:",
    ]
    for lang in LANGUAGE_CATALOG:
        lines.append(f"  \"{lang['name']}\":")
        lines.append(f"    iso: {lang['iso']}")
        lines.append("    scripts: [" + ", ".join(lang["scripts"]) + "]")
        lines.append(f"    region: {lang['region']}")
        lines.append(f"    rtl: {str(lang['rtl']).lower()}")
        lines.append(f"    low_resource: {str(lang['low_resource']).lower()}")
        min_conf = 0.93 if not lang["low_resource"] else 0.88
        lines.append(f"    min_confidence_auto_accept: {min_conf}")

    lines += [
        "",
        "output_scripts:",
        "  - Native",
        "  - Roman/Latin transliteration",
        "  - English gloss",
        "",
        "confidence:",
        f"  word_low_threshold: {CONFIG['confidence']['word_low_threshold']}",
        f"  segment_review_below: {CONFIG['confidence']['flag_avg_confidence_below']}",
        f"  auto_accept_requires_consent: true",
        "",
        "normalisation:",
        "  unicode_form: NFC",
        "  numbers: ascii_digits_no_thousands_separator",
        "  punctuation: fixed_minimal_set",
        "",
        "retention:",
        f"  audio_days: {CONFIG['retention']['audio_days']}",
        f"  transcript_days: {CONFIG['retention']['transcript_days']}",
    ]
    return "\n".join(lines)


LANGUAGES_YML = generate_languages_yml()


# ======================================================================
# LIVE BROWSER-GENERATED DEMO VIDEO
# ======================================================================

VIDEO_HTML_TEMPLATE = r'''
<div style="font-family:Inter,system-ui,sans-serif;background:#0b1220;color:#e5eefb;padding:14px;border-radius:16px;border:1px solid #1f2937;">
  <canvas id="cv" width="960" height="540" style="display:none;"></canvas>
  <video id="vd" controls autoplay muted playsinline style="width:100%;max-height:560px;border-radius:12px;background:#000;"></video>
  <div style="margin-top:12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;">
    <a id="dn" download="global_code_switch_asr_demo.webm" style="display:none;background:#2563eb;color:white;text-decoration:none;padding:9px 13px;border-radius:9px;font-weight:700;">Download live demo video</a>
    <button id="rg" style="background:#334155;color:white;border:0;padding:9px 13px;border-radius:9px;cursor:pointer;font-weight:600;">Regenerate video</button>
    <span id="st" style="color:#94a3b8;">Generating live demo video…</span>
  </div>
</div>

<script>
(function () {
  var languages = __LANGUAGES__;
  var canvas = document.getElementById('cv');
  var ctx = canvas.getContext('2d');
  var video = document.getElementById('vd');
  var status = document.getElementById('st');
  var download = document.getElementById('dn');
  var regen = document.getElementById('rg');

  var duration = 9000;
  var startTime = null;
  var recorder = null;
  var chunks = [];
  var isRecording = false;
  var mimeCandidates = [
    'video/webm;codecs=vp9',
    'video/webm;codecs=vp8',
    'video/webm',
    'video/mp4'
  ];

  function pickMime() {
    if (!window.MediaRecorder) return '';
    for (var i = 0; i < mimeCandidates.length; i++) {
      try {
        if (MediaRecorder.isTypeSupported(mimeCandidates[i])) return mimeCandidates[i];
      } catch (e) {}
    }
    return '';
  }

  function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }

  function roundRect(x, y, w, h, r, fill) {
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
    ctx.fill();
  }

  function text(t, x, y, size, color, align) {
    ctx.fillStyle = color || '#e5eefb';
    ctx.font = (size || 16) + 'px Inter, system-ui, sans-serif';
    ctx.textAlign = align || 'left';
    ctx.textBaseline = 'top';
    ctx.fillText(t, x, y);
  }

  function bar(x, y, w, h, pct, color) {
    roundRect(x, y, w, h, h / 2, '#1e293b');
    roundRect(x, y, Math.max(h, w * clamp(pct, 0, 1)), h, h / 2, color);
  }

  function draw(p) {
    var grad = ctx.createLinearGradient(0, 0, 960, 540);
    grad.addColorStop(0, '#0b1220');
    grad.addColorStop(1, '#111827');
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, 960, 540);

    text('Global Code-Switched Speech AI Review Console', 34, 22, 25, '#f8fafc');
    text('Consent • Multilingual ASR • Confidence • Diarisation • Review • Evaluation', 34, 55, 14, '#94a3b8');

    var stages = ['Consent', 'Audio prep', 'ASR', 'Diarise', 'Review', 'Eval'];
    for (var i = 0; i < stages.length; i++) {
      var x = 34 + i * 148;
      var y = 92;
      var prog = clamp((p - i * 0.07) * 2.0, 0, 1);
      roundRect(x, y, 132, 88, 12, '#111827');
      ctx.strokeStyle = '#1f2937';
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 0.5, y + 0.5, 131, 87);
      text(stages[i], x + 12, y + 14, 15, '#e5eefb');
      bar(x + 12, y + 44, 108, 10, prog, ['#22c55e', '#38bdf8', '#a855f7', '#f59e0b', '#ef4444', '#14b8a6'][i]);
      text(Math.round(prog * 100) + '%', x + 12, y + 62, 16, ['#22c55e', '#38bdf8', '#a855f7', '#f59e0b', '#ef4444', '#14b8a6'][i]);
    }

    roundRect(34, 205, 560, 180, 14, '#111827');
    text('Live language coverage', 50, 222, 18, '#e5eefb');

    var chipCount = 8;
    var offset = Math.floor(p * languages.length * 2);
    for (var j = 0; j < chipCount; j++) {
      var name = languages[(offset + j) % languages.length];
      var cx = 50 + (j % 4) * 132;
      var cy = 255 + Math.floor(j / 4) * 48;
      var pulse = 0.75 + 0.25 * Math.sin(p * 12.56 + j);
      roundRect(cx, cy, 118, 34, 10, 'rgba(56,189,248,' + (0.10 * pulse) + ')');
      ctx.strokeStyle = 'rgba(56,189,248,' + (0.35 * pulse) + ')';
      ctx.strokeRect(cx + 0.5, cy + 0.5, 117, 33);
      text(name, cx + 10, cy + 9, 13, '#bae6fd');
    }

    text('Scripts: native, Roman/Latin, English gloss', 50, 354, 13, '#64748b');

    roundRect(610, 205, 315, 180, 14, '#111827');
    text('Quality targets', 626, 222, 18, '#e5eefb');

    var metrics = [
      ['Overall WER', '12.1%', '#22c55e'],
      ['Code-switch WER', '19.4%', '#f59e0b'],
      ['Low-conf share', '8.2%', '#38bdf8'],
      ['Consent blocked', '2 calls', '#ef4444']
    ];

    for (var k = 0; k < metrics.length; k++) {
      text(metrics[k][0], 626, 258 + k * 30, 14, '#cbd5e1');
      text(metrics[k][1], 900, 258 + k * 30, 14, metrics[k][2], 'right');
    }

    roundRect(34, 405, 891, 88, 14, '#111827');
    text('Pipeline simulation', 50, 422, 18, '#e5eefb');

    var lanes = [
      ['Urdu / English / Roman Urdu', 0.88, '#22c55e'],
      ['Hindi / English', 0.82, '#38bdf8'],
      ['Mandarin / English', 0.79, '#a855f7'],
      ['Arabic / French / English', 0.74, '#f59e0b']
    ];

    for (var m = 0; m < lanes.length; m++) {
      var yy = 450 + m * 0;
      var xx = 50 + m * 218;
      text(lanes[m][0], xx, 450, 12, '#cbd5e1');
      bar(xx, 470, 190, 10, lanes[m][1] * (0.92 + 0.08 * Math.sin(p * 9.42 + m)), lanes[m][2]);
    }

    ctx.font = '13px Inter, system-ui, sans-serif';
    var ticker = 'Consent enforced • Speaker codes not names • NFC normalisation • 16 kHz mono • VAD segments ≤ 30s • Reviewer corrections logged • Legal review before deployment • Group-level bias checks';
    var tw = ctx.measureText(ticker).width;
    var tx = 960 - ((p * (960 + tw)) % (960 + tw));
    text(ticker, tx, 508, 13, '#64748b');

    ctx.fillStyle = '#1f2937';
    ctx.fillRect(34, 528, 892, 4);
    ctx.fillStyle = '#38bdf8';
    ctx.fillRect(34, 528, 892 * p, 4);
  }

  function animatePreview() {
    startTime = null;
    function loop(ts) {
      if (!startTime) startTime = ts;
      var p = ((ts - startTime) % duration) / duration;
      draw(p);
      requestAnimationFrame(loop);
    }
    requestAnimationFrame(loop);
    status.textContent = 'Animated preview running. Browser video recording is unavailable in this environment.';
  }

  function startRecording() {
    if (isRecording) return;

    if (!window.MediaRecorder || !canvas.captureStream) {
      animatePreview();
      return;
    }

    var mime = pickMime();
    if (!mime) {
      animatePreview();
      return;
    }

    isRecording = true;
    chunks = [];
    startTime = null;
    status.textContent = 'Generating live demo video…';
    download.style.display = 'none';
    video.removeAttribute('src');

    var stream = canvas.captureStream(25);

    try {
      recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 2500000 });
    } catch (e) {
      recorder = new MediaRecorder(stream);
    }

    recorder.ondataavailable = function (e) {
      if (e.data && e.data.size > 0) chunks.push(e.data);
    };

    recorder.onstop = function () {
      var blob = new Blob(chunks, { type: recorder.mimeType || mime });
      var url = URL.createObjectURL(blob);
      video.src = url;
      download.href = url;
      download.style.display = 'inline-block';
      status.textContent = 'Live demo video ready. It is generated in-browser and downloadable.';
      isRecording = false;
    };

    recorder.start();

    function frame(ts) {
      if (!startTime) startTime = ts;
      var p = Math.min((ts - startTime) / duration, 1);
      draw(p);

      if (p < 1) {
        requestAnimationFrame(frame);
      } else {
        setTimeout(function () {
          try {
            if (recorder && recorder.state !== 'inactive') recorder.stop();
          } catch (e) {
            isRecording = false;
          }
        }, 120);
      }
    }

    requestAnimationFrame(frame);
  }

  regen.onclick = startRecording;
  setTimeout(startRecording, 150);
})();
</script>
'''


def build_video_html() -> str:
    video_langs = LANGUAGE_NAMES[:18]
    return VIDEO_HTML_TEMPLATE.replace("__LANGUAGES__", json.dumps(video_langs, ensure_ascii=False))


# ======================================================================
# STREAMLIT APP
# ======================================================================

st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Noto+Sans:wght@400;600;700&family=Noto+Nastaliq+Urdu:wght@400;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
  font-family: Inter, "Noto Sans", system-ui, sans-serif;
}
.urdu, .rtl {
  font-family: "Noto Nastaliq Urdu", "Jameel Noori Nastaleeq", "Noto Sans", serif;
}
.word {
  display: inline-block;
  margin: 0 4px 5px 0;
  padding: 3px 6px;
  border-radius: 7px;
  background: rgba(148, 163, 184, 0.10);
  border: 1px solid rgba(148, 163, 184, 0.12);
}
.word.rtl {
  direction: rtl;
  unicode-bidi: isolate;
  font-size: 1.08rem;
  line-height: 2.0;
}
.low-conf {
  background: #fef3c7 !important;
  border-bottom: 2px dashed #b45309;
  color: #78350f;
  font-weight: 600;
}
.transcript {
  padding: 12px 14px;
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  background: #ffffff;
  min-height: 58px;
}
.small-muted {
  color: #64748b;
  font-size: 0.9rem;
}
.pass { color: #15803d; font-weight: 700; }
.fail { color: #b91c1c; font-weight: 700; }
</style>
""",
    unsafe_allow_html=True,
)


# Session state bootstrap
if "calls" not in st.session_state:
    st.session_state.calls = build_demo_calls()

if "records" not in st.session_state:
    st.session_state.records = []

if "sim_events" not in st.session_state:
    st.session_state.sim_events = []

if "auto_sim_done" not in st.session_state:
    st.session_state.auto_sim_done = False

if "selected_call" not in st.session_state:
    st.session_state.selected_call = st.session_state.calls[0]["call_id"]

if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = None

if "last_report" not in st.session_state:
    st.session_state.last_report = None

if "last_prep" not in st.session_state:
    st.session_state.last_prep = []

if "last_error" not in st.session_state:
    st.session_state.last_error = None

if "test_results" not in st.session_state:
    st.session_state.test_results = None

if "language_filter" not in st.session_state:
    st.session_state.language_filter = [x for x in DEFAULT_VISIBLE_LANGUAGES if x in LANGUAGE_NAMES]


# Sidebar
st.sidebar.title("Control room")

language_filter = st.sidebar.multiselect(
    "Languages to monitor",
    LANGUAGE_NAMES,
    default=st.session_state.language_filter,
    key="language_filter_widget",
)
if not language_filter:
    language_filter = LANGUAGE_NAMES
else:
    st.session_state.language_filter = language_filter

script_label = st.sidebar.selectbox(
    "Output script mode",
    ["Auto (native script)", "Roman / Latin transliteration", "English gloss"],
    index=0,
)
script_mode = {"Auto (native script)": "native", "Roman / Latin transliteration": "roman", "English gloss": "english"}[script_label]

min_auto = st.sidebar.slider(
    "Minimum confidence for automatic acceptance",
    0.50, 1.00, float(CONFIG["confidence"]["auto_accept_min"]), step=0.01
)
low_thresh = st.sidebar.slider(
    "Low-confidence word threshold",
    0.30, 1.00, float(CONFIG["confidence"]["word_low_threshold"]), step=0.01
)
flag_avg = st.sidebar.slider(
    "Flag recording if average confidence below",
    0.50, 1.00, float(CONFIG["confidence"]["flag_avg_confidence_below"]), step=0.01
)

CONFIG["confidence"]["auto_accept_min"] = float(min_auto)
CONFIG["confidence"]["word_low_threshold"] = float(low_thresh)
CONFIG["confidence"]["flag_avg_confidence_below"] = float(flag_avg)

audio_days = st.sidebar.number_input(
    "Audio retention days", min_value=1, max_value=3650, value=int(CONFIG["retention"]["audio_days"])
)
transcript_days = st.sidebar.number_input(
    "Transcript retention days", min_value=1, max_value=3650, value=int(CONFIG["retention"]["transcript_days"])
)
CONFIG["retention"]["audio_days"] = int(audio_days)
CONFIG["retention"]["transcript_days"] = int(transcript_days)

reviewer_id = st.sidebar.text_input("Reviewer ID", "reviewer-demo")
consent_required = st.sidebar.checkbox("Enforce recorded consent", value=True)

visible_calls = [
    c for c in st.session_state.calls
    if c["primary_language"] in language_filter
    or any(l in language_filter for l in c.get("secondary_languages", []))
]
if not visible_calls:
    visible_calls = st.session_state.calls

call_ids = [c["call_id"] for c in visible_calls]
if st.session_state.selected_call not in call_ids:
    st.session_state.selected_call = call_ids[0]

selected_index = call_ids.index(st.session_state.selected_call)
selected_id = st.sidebar.selectbox("Selected call", call_ids, index=selected_index)
st.session_state.selected_call = selected_id

base_call = next(c for c in st.session_state.calls if c["call_id"] == selected_id)
call = copy.deepcopy(base_call)

consent_value = st.sidebar.text_input(
    "Consent reference for selected call",
    value=base_call.get("consent_ref", ""),
    key=f"consent_{selected_id}",
)
base_call["consent_ref"] = consent_value.strip()
call["consent_ref"] = base_call["consent_ref"]

notice_given = st.sidebar.checkbox(
    "Notice given for selected call",
    value=bool(base_call.get("notice_given", True)),
    key=f"notice_{selected_id}",
)
base_call["notice_given"] = bool(notice_given)
call["notice_given"] = bool(notice_given)

st.sidebar.markdown("---")

if st.sidebar.button("Reset demo"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# Header
st.title(APP_TITLE)
st.caption(APP_SUBTITLE)


# Automatic bootstrap simulation
def simulate_event(i: int) -> Dict[str, Any]:
    calls = st.session_state.calls
    c = calls[i % len(calls)]
    rep = call_report(c)
    consent_ok = has_valid_consent(c)

    if not consent_ok:
        action = "blocked: consent gate"
    elif rep["flag_for_human_review"]:
        action = "queued: full human review"
    elif rep["auto_accept_eligible"]:
        action = "auto-accept candidate"
    else:
        action = "queued: spot check"

    return {
        "tick": i + 1,
        "time": now_iso(),
        "call_id": c["call_id"],
        "primary_language": c.get("primary_language", ""),
        "consent_ok": consent_ok,
        "avg_confidence": rep["average_word_confidence"],
        "low_conf_share": rep["low_confidence_word_share"],
        "code_switch_share": rep["code_switched_segment_share"],
        "action": action,
    }


if not st.session_state.auto_sim_done:
    st.session_state.auto_sim_done = True
    st.markdown("### Automatic demo bootstrap")
    prog = st.progress(0.0, text="Loading realistic multilingual demo data…")
    box = st.empty()
    ticks = 10

    for i in range(ticks):
        ev = simulate_event(i)
        st.session_state.sim_events.append(ev)
        df = pd.DataFrame(st.session_state.sim_events)
        box.dataframe(df, use_container_width=True)
        prog.progress((i + 1) / ticks, text=f"Simulating consent-gated transcription tick {i + 1}/{ticks}…")
        time.sleep(0.22)

    prog.empty()
    box.empty()
    st.success("Demo data loaded and simulation started. Use the tabs below to interact.")


# Tabs
tabs = st.tabs(
    [
        "🎥 Live demo video",
        "🌍 Global languages",
        "📡 Simulation",
        "🔐 Consent & governance",
        "🗂️ Data & split",
        "🎧 Audio prep",
        "📝 Transcription",
        "👥 Diarisation",
        "🧑‍⚖️ Review",
        "📊 Evaluation",
        "✅ Tests",
        "📚 Docs",
    ]
)


# Live video tab
with tabs[0]:
    st.header("Generated live demo video")
    st.markdown(
        "The video below is generated immediately in your browser from an animated canvas. "
        "When recording finishes, use the download link to save a `.webm` video file."
    )
    st.components.v1.html(build_video_html(), height=760)


# Global languages tab
with tabs[1]:
    st.header("Global language coverage")

    coverage_rows = []
    for lang in LANGUAGE_CATALOG:
        coverage_rows.append(
            {
                "Language": lang["name"],
                "ISO": lang["iso"],
                "Scripts": ", ".join(lang["scripts"]),
                "Region": lang["region"],
                "Speakers (millions)": lang["speakers_millions"],
                "RTL": lang["rtl"],
                "Low-resource": lang["low_resource"],
                "Monitored": lang["name"] in language_filter,
            }
        )
    coverage_df = pd.DataFrame(coverage_rows)
    st.dataframe(coverage_df, use_container_width=True, hide_index=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Languages in catalog", len(LANGUAGE_CATALOG))
    c2.metric("Languages monitored", len([x for x in language_filter if x in LANGUAGE_NAMES]))
    c3.metric("RTL languages", sum(1 for x in LANGUAGE_CATALOG if x["rtl"]))
    c4.metric("Low-resource flags", sum(1 for x in LANGUAGE_CATALOG if x["low_resource"]))

    st.subheader("Top monitored languages by simulated speaker volume")
    mon_df = coverage_df[coverage_df["Monitored"]].copy()
    if not mon_df.empty:
        chart_df = mon_df.set_index("Language")["Speakers (millions)"].sort_values(ascending=False).head(15)
        st.bar_chart(chart_df)


# Simulation tab
with tabs[2]:
    st.header("Live incoming-call simulation")

    col1, col2, col3, col4, col5 = st.columns(5)
    total = len(st.session_state.sim_events)
    blocked = sum(1 for e in st.session_state.sim_events if not e["consent_ok"])
    review = sum(1 for e in st.session_state.sim_events if "human review" in e["action"])
    auto = sum(1 for e in st.session_state.sim_events if e["action"] == "auto-accept candidate")
    code = sum(1 for e in st.session_state.sim_events if e["code_switch_share"] > 0)

    col1.metric("Simulated ticks", total)
    col2.metric("Consent blocked", blocked)
    col3.metric("Human review", review)
    col4.metric("Auto-accept candidates", auto)
    col5.metric("Code-switched calls", code)

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Run one more simulation tick"):
            st.session_state.sim_events.append(simulate_event(len(st.session_state.sim_events)))
            st.rerun()
    with b2:
        if st.button("Clear simulation events"):
            st.session_state.sim_events = []
            st.rerun()

    if st.session_state.sim_events:
        st.dataframe(pd.DataFrame(st.session_state.sim_events), use_container_width=True, hide_index=True)
    else:
        st.info("No simulation events yet.")

    st.subheader("Recording-level confidence report")
    report_rows = [call_report(c) for c in visible_calls]
    st.dataframe(pd.DataFrame(report_rows), use_container_width=True, hide_index=True)


# Consent and governance tab
with tabs[3]:
    st.header("Consent gate and governance")

    st.markdown(SCOPE_MD)
    st.download_button("Download scope.md", SCOPE_MD, file_name="scope.md", mime="text/markdown", key="dl_scope_tab")

    st.markdown(CONSENT_LAW_MD)
    st.download_button(
        "Download consent_and_law.md",
        CONSENT_LAW_MD,
        file_name="consent_and_law.md",
        mime="text/markdown",
        key="dl_consent_tab",
    )

    st.subheader("Selected call metadata")
    meta = {
        "call_id": call["call_id"],
        "consent_ref": call.get("consent_ref", ""),
        "notice_given": bool(call.get("notice_given", True)),
        "primary_language": call.get("primary_language"),
        "secondary_languages": call.get("secondary_languages"),
        "region": call.get("region"),
        "accent": call.get("accent"),
        "age_band": call.get("age_band"),
        "caller_gender": call.get("caller_gender"),
        "phone_quality": call.get("phone_quality"),
        "background_noise": call.get("background_noise"),
        "speaker_roles": call.get("speaker_roles"),
    }
    st.json(meta)

    if st.button("Verify consent for selected call"):
        try:
            ensure_consent(call, consent_required)
            st.success("Consent reference accepted. Transcription may proceed.")
        except ConsentError as e:
            st.error(str(e))

    st.subheader("Retention calculator")
    today = dt.datetime.now(dt.timezone.utc).date()
    audio_delete = today + dt.timedelta(days=int(CONFIG["retention"]["audio_days"]))
    transcript_delete = today + dt.timedelta(days=int(CONFIG["retention"]["transcript_days"]))
    st.write(f"Audio deletion due: **{audio_delete.isoformat()}**")
    st.write(f"Transcript deletion due: **{transcript_delete.isoformat()}**")


# Data and split tab
with tabs[4]:
    st.header("Data collection, speaker codes, and leakage-safe splits")

    st.markdown(
        """
        The demo stores speaker codes instead of names. Train/test splits are checked for speaker and call leakage.
        A second transcriber is assigned to a configurable share of segments, and disagreements are recorded.
        """
    )

    splits = build_splits(st.session_state.calls)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train segments", len(splits["good_train"]))
    c2.metric("Test segments", len(splits["good_test"]))
    c3.metric("Train calls", len(splits["train_calls"]))
    c4.metric("Test calls", len(splits["test_calls"]))

    st.subheader("Split validation")
    st.write("Clean split errors:", splits["good_errors"] or "None")
    st.write("Intentional bad split errors:", splits["bad_errors"])

    st.subheader("Demographic coverage in demo data")
    demo_df = pd.DataFrame(
        [
            {
                "call_id": c["call_id"],
                "primary_language": c["primary_language"],
                "region": c["region"],
                "accent": c["accent"],
                "age_band": c["age_band"],
                "caller_gender": c["caller_gender"],
                "phone_quality": c["phone_quality"],
                "background_noise": c["background_noise"],
                "consent_valid": has_valid_consent(c),
            }
            for c in st.session_state.calls
        ]
    )
    st.dataframe(demo_df, use_container_width=True, hide_index=True)

    st.subheader("Second-transcriber disagreements")
    dis_rows = []
    for c in st.session_state.calls:
        for s in c.get("segments", []):
            for d in s.get("disagreements", []):
                dis_rows.append(
                    {
                        "call_id": c["call_id"],
                        "segment_id": s["segment_id"],
                        "word_index": d["word_index"],
                        "annotator_a": d["annotator_a"],
                        "annotator_b": d["annotator_b"],
                        "resolved": d["resolved"],
                    }
                )
    if dis_rows:
        st.dataframe(pd.DataFrame(dis_rows), use_container_width=True, hide_index=True)
    else:
        st.info("No disagreements recorded in current demo data.")


# Audio prep tab
with tabs[5]:
    st.header("Audio preparation and voice activity detection")

    st.markdown(
        """
        The pipeline converts input to 16 kHz mono, estimates SNR, applies only light noise reduction when needed,
        and splits speech into segments of up to 30 seconds cut at silences.
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Target sample rate", "16 kHz")
    c2.metric("Max segment", f"{CONFIG['vad']['max_segment_s']}s")
    c3.metric("SNR threshold", f"{CONFIG['vad']['snr_threshold_db']} dB")
    c4.metric("Noise reduction", "Light only")

    st.subheader("Synthetic two-burst VAD test")
    test_audio = synth_test_audio_array()
    test_segments = vad_segments_from_array(test_audio, 16000)
    st.write("Detected segments:", test_segments)
    st.audio(float_to_wav(test_audio, 16000), format="audio/wav")

    st.subheader("Optional WAV upload")
    uploaded = st.file_uploader("Upload a 16-bit mono WAV file to test preprocessing", type=["wav"])
    if uploaded is not None:
        try:
            prep = prepare_audio_from_bytes(uploaded.getvalue())
            st.audio(prep["wav"], format="audio/wav")
            st.write(f"Estimated SNR: **{prep['snr_db']} dB**")
            st.write(f"Noise reduced: **{prep['noise_reduced']}**")
            st.write("VAD segments:")
            st.dataframe(pd.DataFrame(prep["segments"], columns=["start", "end"]), use_container_width=True)
        except Exception as e:
            st.error(f"Could not process uploaded file: {e}")


# Transcription tab
with tabs[6]:
    st.header("Transcription console")

    if st.button("Transcribe selected call"):
        try:
            transcript, report, prep = run_transcription(call, script_mode, consent_required)
            st.session_state.last_transcript = transcript
            st.session_state.last_report = report
            st.session_state.last_prep = prep
            st.session_state.last_error = None
        except ConsentError as e:
            st.session_state.last_error = str(e)
            st.session_state.last_transcript = None
            st.session_state.last_report = None
            st.session_state.last_prep = []

    if st.session_state.last_error:
        st.error(st.session_state.last_error)
        st.warning("Transcription did not start because the consent gate stopped it.")

    if st.session_state.last_transcript:
        st.success(
            f"Transcript generated for {st.session_state.last_transcript['call_id']} "
            f"in {st.session_state.last_transcript['script_mode']} script mode."
        )

        rep = st.session_state.last_report
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Average word confidence", f"{rep['average_word_confidence']:.3f}")
        c2.metric("Low-confidence share", f"{rep['low_confidence_word_share'] * 100:.1f}%")
        c3.metric("Code-switched segments", f"{rep['code_switched_segment_share'] * 100:.1f}%")
        c4.metric("Human review flag", "Yes" if rep["flag_for_human_review"] else "No")

        st.subheader("Audio preparation check")
        if st.session_state.last_prep:
            st.dataframe(pd.DataFrame(st.session_state.last_prep), use_container_width=True, hide_index=True)

        st.subheader("Readable transcript")
        readable = readable_transcript(call, script_mode)
        st.code(readable, language="text")

        st.subheader("Highlighted transcript")
        for seg in sorted(call["segments"], key=lambda s: s["start"]):
            role = seg.get("speaker_role", generic_role(seg["speaker_label"], call.get("speaker_roles")))
            st.markdown(
                f"**{esc(role)}** · {esc(seg['language'])} · "
                f"confidence `{seg['confidence']:.2f}` · "
                f"`{seg['start']:.1f}s–{seg['end']:.1f}s`"
            )
            st.markdown(
                f"<div class='transcript'>{html_transcript_segment(seg, script_mode)}</div>",
                unsafe_allow_html=True,
            )
            st.audio(segment_audio_bytes(seg), format="audio/wav")
            st.markdown("<div class='small-muted'>Synthetic segment audio for demo playback.</div>", unsafe_allow_html=True)

        with st.expander("JSON transcript"):
            st.json(st.session_state.last_transcript)

        d1, d2 = st.columns(2)
        with d1:
            st.download_button(
                "Download transcript JSON",
                json.dumps(st.session_state.last_transcript, ensure_ascii=False, indent=2),
                file_name=f"{call['call_id']}_transcript.json",
                mime="application/json",
                key="dl_json",
            )
        with d2:
            st.download_button(
                "Download readable transcript",
                readable,
                file_name=f"{call['call_id']}_transcript.txt",
                mime="text/plain",
                key="dl_txt",
            )
    else:
        st.info("Press “Transcribe selected call” to run the consent-gated transcription simulation.")


# Diarisation tab
with tabs[7]:
    st.header("Speaker diarisation and role mapping")

    st.markdown(
        """
        Speakers are labelled generically unless call metadata supplies roles.
        Word timestamps are merged with speaker turns.
        """
    )

    turn_rows = []
    for t in call.get("speaker_turns", []):
        turn_rows.append(
            {
                "speaker_label": t["speaker"],
                "role": t["role"],
                "start": t["start"],
                "end": t["end"],
                "metadata_role_available": bool(call.get("speaker_roles") and t["speaker"] in call.get("speaker_roles", {})),
            }
        )
    st.dataframe(pd.DataFrame(turn_rows), use_container_width=True, hide_index=True)

    st.subheader("Word-level speaker assignment sample")
    word_rows = []
    for seg in call.get("segments", [])[:2]:
        for w in seg.get("words", [])[:8]:
            word_rows.append(
                {
                    "segment_id": seg["segment_id"],
                    "text": word_display(w, script_mode),
                    "language": w.get("lang", ""),
                    "speaker_label": w.get("speaker", seg["speaker_label"]),
                    "role": w.get("role", generic_role(seg["speaker_label"], call.get("speaker_roles"))),
                    "start": w.get("start", 0),
                    "end": w.get("end", 0),
                    "confidence": w.get("conf", 0),
                }
            )
    st.dataframe(pd.DataFrame(word_rows), use_container_width=True, hide_index=True)


# Review tab
with tabs[8]:
    st.header("Human review workbench")

    st.markdown(
        "Reviewers can listen to a segment, inspect highlighted low-confidence words, "
        "and save corrections with reviewer ID and timestamp."
    )

    seg_options = [
        f"{s['segment_id']} · {s.get('speaker_role', s['speaker_label'])} · {s['language']} · {s['start']:.1f}s"
        for s in call["segments"]
    ]
    seg_idx = st.selectbox(
        "Select segment",
        range(len(call["segments"])),
        format_func=lambda i: seg_options[i],
        key=f"seg_select_{call['call_id']}",
    )
    seg = call["segments"][seg_idx]

    st.audio(segment_audio_bytes(seg), format="audio/wav")

    st.markdown(
        f"<div class='transcript'>{html_transcript_segment(seg, script_mode)}</div>",
        unsafe_allow_html=True,
    )

    if seg.get("disagreements"):
        st.subheader("Recorded second-transcriber disagreements")
        st.dataframe(pd.DataFrame(seg["disagreements"]), use_container_width=True, hide_index=True)

    st.subheader("Word-level correction")
    word_labels = [
        f"{i}: {word_display(w, script_mode)} · conf {w['conf']:.2f} · {w['lang']}"
        for i, w in enumerate(seg["words"])
    ]
    if word_labels:
        wi = st.selectbox(
            "Select word",
            range(len(seg["words"])),
            format_func=lambda i: word_labels[i],
            key=f"word_select_{seg['segment_id']}",
        )
        current = word_display(seg["words"][wi], script_mode)
        corrected = st.text_input(
            "Corrected word",
            value=current,
            key=f"corrected_word_{seg['segment_id']}_{wi}",
        )

        if st.button("Save word correction"):
            record = {
                "timestamp": now_iso(),
                "reviewer_id": reviewer_id,
                "call_id": call["call_id"],
                "segment_id": seg["segment_id"],
                "word_index": wi,
                "original": current,
                "corrected": normalize_text(corrected),
                "type": "word_correction",
            }
            st.session_state.records.append(record)
            st.success("Word correction saved.")
    else:
        st.warning("This segment has no words.")

    st.subheader("Segment-level correction")
    segment_text = " ".join(word_display(w, script_mode) for w in seg["words"])
    corrected_segment = st.text_area(
        "Corrected segment transcript",
        value=segment_text,
        height=100,
        key=f"corrected_segment_{seg['segment_id']}",
    )

    if st.button("Save segment version"):
        version = 1 + sum(
            1
            for r in st.session_state.records
            if r.get("call_id") == call["call_id"] and r.get("type") == "segment_version"
        )
        record = {
            "timestamp": now_iso(),
            "reviewer_id": reviewer_id,
            "call_id": call["call_id"],
            "segment_id": seg["segment_id"],
            "version": version,
            "original": segment_text,
            "corrected": normalize_text(corrected_segment),
            "type": "segment_version",
        }
        st.session_state.records.append(record)
        st.success("Segment version saved.")

    st.subheader("Correction history")
    if st.session_state.records:
        rdf = pd.DataFrame(st.session_state.records)
        st.dataframe(rdf, use_container_width=True, hide_index=True)
        st.download_button(
            "Download correction records JSON",
            json.dumps(st.session_state.records, ensure_ascii=False, indent=2),
            file_name="review_records.json",
            mime="application/json",
            key="dl_records",
        )
    else:
        st.info("No corrections saved yet.")


# Evaluation tab
with tabs[9]:
    st.header("Held-out evaluation dashboard")

    eval_df = build_eval_dataframe(language_filter)
    report_md, summary = build_eval_report(eval_df)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Held-out calls", summary.get("held_out_calls", 0))
    c2.metric("Overall WER", f"{summary.get('overall_wer_pct', 0)}%")
    c3.metric("Code-switched WER", f"{summary.get('code_switched_wer_pct', 0)}%")
    c4.metric("Low-resource WER", f"{summary.get('low_resource_wer_pct', 0)}%")
    c5.metric("Worst group gap", f"{summary.get('max_group_gap_points', 0)} pts")

    st.markdown(
        f"""
        **Targets**

        - Overall WER below 15%: {'✅ PASS' if summary.get('pass_overall') else '❌ FAIL'}
        - Code-switched WER below 25%: {'✅ PASS' if summary.get('pass_code_switched') else '❌ FAIL'}
        - No group more than 5 points worse than average: {'✅ PASS' if summary.get('pass_group_gap') else '❌ FAIL'}
        - Low-resource WER below 20%: {'✅ PASS' if summary.get('pass_low_resource') else '❌ FAIL'}
        """
    )

    st.subheader("Metrics by language")
    lang_df = (
        eval_df.groupby("language")[["wer", "cer"]]
        .mean()
        .mul(100)
        .round(2)
        .reset_index()
        .rename(columns={"wer": "WER (%)", "cer": "CER (%)"})
        .sort_values("WER (%)", ascending=False)
    )
    st.dataframe(lang_df, use_container_width=True, hide_index=True)

    if not lang_df.empty:
        st.bar_chart(lang_df.set_index("language")["WER (%)"].head(15))

    st.subheader("Metrics by speaker group")
    group_df = (
        eval_df.groupby("speaker_group")[["wer", "cer", "inter_annotator_agreement"]]
        .mean()
        .reset_index()
    )
    group_df["WER (%)"] = (group_df["wer"] * 100).round(2)
    group_df["CER (%)"] = (group_df["cer"] * 100).round(2)
    group_df["Agreement"] = group_df["inter_annotator_agreement"].round(3)
    st.dataframe(group_df[["speaker_group", "WER (%)", "CER (%)", "Agreement"]], use_container_width=True, hide_index=True)

    st.subheader("Metrics by condition")
    cond_df = (
        eval_df.groupby(["code_switched", "phone_quality", "background_noise"])[["wer"]]
        .mean()
        .reset_index()
    )
    cond_df["WER (%)"] = (cond_df["wer"] * 100).round(2)
    st.dataframe(cond_df[["code_switched", "phone_quality", "background_noise", "WER (%)"]], use_container_width=True, hide_index=True)

    st.subheader("Annotation protocol")
    st.markdown(
        "- Two native speakers transcribe each held-out call.\n"
        "- A third resolver settles disagreements.\n"
        "- WER and CER are computed after NFC normalisation, ASCII digit rules, and fixed punctuation rules.\n"
        "- Results are reported separately for each language, script, code-switch status, speaker group, phone quality, and noise condition."
    )

    st.subheader("Report")
    st.code(report_md, language="markdown")

    saved = False
    try:
        os.makedirs("reports", exist_ok=True)
        with open(os.path.join("reports", "asr_eval.md"), "w", encoding="utf-8") as f:
            f.write(report_md)
        saved = True
    except Exception:
        saved = False

    if saved:
        st.caption("Report also written to `reports/asr_eval.md` when filesystem access allows.")

    st.download_button(
        "Download asr_eval.md",
        report_md,
        file_name="asr_eval.md",
        mime="text/markdown",
        key="dl_eval",
    )


# Tests tab
with tabs[10]:
    st.header("Internal tests and acceptance checklist")

    if st.button("Run / re-run tests") or st.session_state.test_results is None:
        with st.spinner("Running internal tests…"):
            st.session_state.test_results = run_internal_tests()

    tr = st.session_state.test_results
    st.dataframe(tr, use_container_width=True, hide_index=True)

    pass_count = int((tr["Result"] == "PASS").sum())
    fail_count = int((tr["Result"] == "FAIL").sum())
    st.markdown(f"**Passed:** {pass_count} · **Failed:** {fail_count}")

    st.subheader("Acceptance checklist")
    st.markdown(
        """
        - [x] Transcription does not start without a recorded consent reference, verified by a test.
        - [x] Speakers are labelled generically unless metadata supports a role, verified by a test.
        - [x] Low-confidence words are flagged for review.
        - [x] Error rates are reported for each language and code-switched speech separately.
        - [x] Word error rate is measured by native speakers before transcripts are relied on.
        - [x] Global language coverage includes major world languages and Roman Urdu.
        """
    )

    st.subheader("Risks and mitigations")
    risk_df = pd.DataFrame(
        [
            {
                "Risk": "Recording people without proper consent",
                "Mitigation": "Consent references, notices, and legal review before deployment",
            },
            {
                "Risk": "Transcription errors changing the meaning of important statements",
                "Mitigation": "Confidence flags, review tool, and rule that transcripts are checked before formal use",
            },
            {
                "Risk": "Bias in accuracy across accents, dialects, or low-resource languages",
                "Mitigation": "Group-level evaluation, low-resource monitoring, and wider training data",
            },
            {
                "Risk": "Code-switched speech being mishandled",
                "Mitigation": "Separate code-switch metrics, mixed-language training simulation, and human review",
            },
        ]
    )
    st.dataframe(risk_df, use_container_width=True, hide_index=True)


# Docs tab
with tabs[11]:
    st.header("Project documents and config")

    st.subheader("docs/scope.md")
    st.markdown(SCOPE_MD)
    st.download_button("Download scope.md", SCOPE_MD, file_name="scope.md", mime="text/markdown", key="dl_scope_docs")

    st.subheader("docs/consent_and_law.md")
    st.markdown(CONSENT_LAW_MD)
    st.download_button(
        "Download consent_and_law.md",
        CONSENT_LAW_MD,
        file_name="consent_and_law.md",
        mime="text/markdown",
        key="dl_consent_docs",
    )

    st.subheader("config/languages.yml")
    st.code(LANGUAGES_YML, language="yaml")
    st.download_button(
        "Download languages.yml",
        LANGUAGES_YML,
        file_name="languages.yml",
        mime="text/yaml",
        key="dl_lang_docs",
    )

    st.subheader("config/load_settings.py behaviour")
    st.markdown(
        """
        In this single-file demo, the consent gate is implemented by `ensure_consent()` and used by
        `run_transcription()`. If a recording has no consent reference, or if notice was not given,
        transcription is refused and the UI shows the reason.
        """
    )