# Copyright (c) 2026 Jessica Wang. All Rights Reserved.
# See LICENSE file for terms. Unauthorised use, copying, or distribution is prohibited.

"""
Contracts whose underlying is not a crypto asset — frozen 2026-10-12 (see PREREGISTRATION.md §2).

How the list was built, without looking at returns:
  - TRADFI and COMMODITY (measured): contracts first traded on or after 2025-06-01 whose median
    weekend hourly volume is below 60% of their weekday median. Every crypto token listed in that
    window sits at 0.64 or above; every stock, ETF, commodity and pre-IPO contract at 0.53 or below,
    with volume concentrated in exchange hours. (BTC scores 0.55 over its full history, which is why
    the rule only applies to recent listings.)
  - TRADFI (named): contracts with under two weeks of trading in the sample, named after listed or
    pre-IPO companies and listed alongside them.
  - INDEX, STABLECOIN, and the tokenized-gold COMMODITY entries: named.
"""

INDEX = frozenset({"DEFIUSDT", "BTCDOMUSDT", "FOOTBALLUSDT", "BLUEBIRDUSDT"})

STABLECOIN = frozenset({"USDCUSDT"})

COMMODITY = frozenset({
    "BZUSDT", "CLUSDT", "COPPERUSDT", "NATGASUSDT", "PAXGUSDT", "XAGUSDT", "XAUTUSDT", "XAUUSDT",
    "XPDUSDT", "XPTUSDT",
})

TRADFI = frozenset({
    # measured: weekend volume ratio < 0.6, first traded 2025-06 or later
    "AAOIUSDT", "AAPLUSDT", "ADBEUSDT", "ALABUSDT", "AMATUSDT", "AMDUSDT", "AMZNUSDT", "ANTHROPICUSDT",
    "APPUSDT", "ARMUSDT", "ASMLUSDT", "ASTSUSDT", "AVGOUSDT", "AXTIUSDT", "BABAUSDT", "BBXUSDT", "BEUSDT",
    "BITOUSDT", "BMNRUSDT", "BNCUSDT", "BOTUSDT", "BRKBUSDT", "BSPUSDT", "BXUSDT", "CATUSDT", "CBRSUSDT",
    "CIENUSDT", "COHRUSDT", "COINUSDT", "COSTUSDT", "CRCLUSDT", "CRDOUSDT", "CRMUSDT", "CRWDUSDT",
    "CRWVUSDT", "CSCOUSDT", "CSOPSAMSUNG2LUSDT", "CSOPSKHYNIX2LUSDT", "DELLUSDT", "DISUSDT", "DKNGUSDT",
    "DRAMUSDT", "EBAYUSDT", "EWJUSDT", "EWTUSDT", "EWYUSDT", "EWZUSDT", "FLEXUSDT", "FLNCUSDT", "FWDIUSDT",
    "GDXUSDT", "GEVUSDT", "GIGADEVUSDT", "GLWUSDT", "GMEUSDT", "GOOGLUSDT", "GSUSDT", "HANMIUSDT", "HDUSDT",
    "HIMSUSDT", "HK0700USDT", "HK1810USDT", "HOODUSDT", "HPEUSDT", "HYUNDAIUSDT", "IBMUSDT", "INTCUSDT",
    "INTWUSDT", "IRENUSDT", "IWMUSDT", "JPMUSDT", "KLACUSDT", "KODEX200USDT", "KORUUSDT", "KOUSDT",
    "KSTRUSDT", "KUAISHOUUSDT", "LGELECTRONICSUSDT", "LITEUSDT", "LLYUSDT", "LRCXUSDT", "LYTEUSDT",
    "MEITUANUSDT", "METAUSDT", "MINIMAXUSDT", "MRVLUSDT", "MSFTUSDT", "MSTRUSDT", "MUUSDT", "MUUUSDT",
    "MVLLUSDT", "NAVERUSDT", "NBISUSDT", "NETUSDT", "NFLXUSDT", "NOKUSDT", "NOWUSDT", "NVDAUSDT", "NVOUSDT",
    "ONDSUSDT", "OPENAIUSDT", "ORCLUSDT", "PANWUSDT", "PAYPUSDT", "PENGUSDT", "PLTRUSDT", "POPMARTUSDT",
    "PYPLUSDT", "QCOMUSDT", "QNTXUSDT", "QQQUSDT", "RDDTUSDT", "RIVNUSDT", "RKLBUSDT", "SAMSUNGEMUSDT",
    "SAMSUNGUSDT", "SHAZUSDT", "SHOPUSDT", "SKHYNIXUSDT", "SKHYUSDT", "SMCIUSDT", "SMHUSDT", "SNDKUSDT",
    "SNOWUSDT", "SNXXUSDT", "SOFIUSDT", "SONYUSDT", "SOXLUSDT", "SOXSUSDT", "SPCXUSDT", "SPYUSDT",
    "SQQQUSDT", "STRCUSDT", "STXXUSDT", "TBTUSDT", "TENCENTUSDT", "TERUSDT", "TMFUSDT", "TQQQUSDT",
    "TSLAUSDT", "TSMUSDT", "TTWOUSDT", "TXNUSDT", "TZAUSDT", "UBERUSDT", "URNMUSDT", "USARUSDT", "UVXYUSDT",
    "VRTUSDT", "VSTUSDT", "VUSDT", "WDCUSDT", "WENUSDT", "WMTUSDT", "XBIUSDT", "XLEUSDT", "ZHIPUUSDT",
    "ZHONGJIUSDT", "ZMUSDT",
    # named: under two weeks of trading in the sample
    "CXMTUSDT", "DJTUSDT", "IONQUSDT", "MARAUSDT", "MRKUSDT", "MRNAUSDT", "PDDUSDT", "RAMUSDT", "SKDDUSDT",
    "SKUUUSDT", "TEMUSDT", "UNITREEUSDT",
})

NON_CRYPTO = INDEX | STABLECOIN | COMMODITY | TRADFI


def is_crypto(symbol):
    return symbol not in NON_CRYPTO
