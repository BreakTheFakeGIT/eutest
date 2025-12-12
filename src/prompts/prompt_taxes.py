from typing import Dict, List, Literal

Area = Literal['vat','pit','cit','pcc','psd','akcyza',
               'op','gry','malpki','spw','pt','pkop',
               'spdet','fin','cukier','wip','globe','nip','inne'
               ]

# ---------- MAIN ----------
questions_main: List[str] = [
    "Wymień 10 słów kluczowych w zakresie podatku.",
    "W 5 punktach, wymień artykuły, przepisy wskazane w tekście.",
    "W 5 punktach, podsumuj fragment tekstu."
]

# ---------- VAT ----------
prompt_vat: str = "Jesteś ekspertem podatkowym w zakresie podatku VAT."
questions_vat: List[str] = questions_main
PROMPT_VAT: Dict[str, List[str] | str] = {"prompt": prompt_vat, "questions": questions_vat}

# ---------- PIT ----------
prompt_pit: str = "Jesteś ekspertem podatkowym w zakresie PIT."
questions_pit: List[str] = questions_main
PROMPT_PIT: Dict[str, List[str] | str] = {"prompt": prompt_pit, "questions": questions_pit}

# ---------- CIT ----------
prompt_cit: str = "Jesteś ekspertem podatkowym w zakresie CIT."
questions_cit: List[str] = questions_main
PROMPT_CIT: Dict[str, List[str] | str] = {"prompt": prompt_cit, "questions": questions_cit}

# ---------- PCC ----------
prompt_pcc = "Jesteś ekspertem podatkowym specjalizującym się w podatku od czynności cywilnoprawnych (PCC)."
questions_pcc = questions_main
PROMPT_PCC = {"prompt": prompt_pcc, "questions": questions_pcc}

# ---------- PSD ----------
prompt_psd = "Jesteś ekspertem podatkowym specjalizującym się w podatku od spadków i darowizn."
questions_psd = questions_main
PROMPT_PSD = {"prompt": prompt_psd, "questions": questions_psd}

# ---------- AKCYZA ----------
prompt_akcyza = "Jesteś ekspertem podatkowym specjalizującym się w podatku akcyzowym."
questions_akcyza = questions_main
PROMPT_AKCYZA = {"prompt": prompt_akcyza, "questions": questions_akcyza}

# ---------- ORDYNACJA PODATKOWA ----------
prompt_op = "Jesteś ekspertem podatkowym specjalizującym się w ordynacji podatkowej."
questions_op = questions_main
PROMPT_OP = {"prompt": prompt_op, "questions": questions_op}

# ---------- GRY HAZARDOWE ----------
prompt_gry = "Jesteś ekspertem podatkowym specjalizującym się w grach hazardowych."
questions_gry = questions_main
PROMPT_GRY = {"prompt": prompt_gry, "questions": questions_gry}

# ---------- MALPKI ----------
prompt_malpki = "Jesteś ekspertem podatkowym specjalizującym się w wychowaniu w trzeźwości i przeciwdziałaniu alkoholizmowi."
questions_malpki = questions_main
PROMPT_MALPKI = {"prompt": prompt_malpki, "questions": questions_malpki}

# ---------- WĘGLOWODORY ----------
prompt_spw = "Jesteś ekspertem podatkowym specjalizującym się w podatku węglowodorowym."
questions_spw = questions_main
PROMPT_SPW = {"prompt": prompt_spw, "questions": questions_spw}

# ---------- TONAŻ ----------
prompt_pt = "Jesteś ekspertem podatkowym specjalizującym się w podatku tonażowym."
questions_pt = questions_main
PROMPT_PT = {"prompt": prompt_pt, "questions": questions_pt}

# ---------- KOPALINY ----------
prompt_pkop = "Jesteś ekspertem podatkowym specjalizującym się w podatku od wydobycia niektórych kopalin."
questions_pkop = questions_main
PROMPT_PKOP = {"prompt": prompt_pkop, "questions": questions_pkop}

# ---------- SPRZEDAŻ DETALICZNA ----------
prompt_spdet = "Jesteś ekspertem podatkowym specjalizującym się w podatku od sprzedaży detalicznej."
questions_spdet = questions_main
PROMPT_SPDET = {"prompt": prompt_spdet, "questions": questions_spdet}

# ---------- FINANSE ----------
prompt_fin = "Jesteś ekspertem podatkowym specjalizującym się w podatku od niektórych instytucji finansowych."
questions_fin = questions_main
PROMPT_FIN = {"prompt": prompt_fin, "questions": questions_fin}

# ---------- CUKIER ----------
prompt_cukier = "Jesteś ekspertem podatkowym specjalizującym się w zdrowiu publicznym."
questions_cukier = questions_main
PROMPT_CUKIER = {"prompt": prompt_cukier, "questions": questions_cukier}

# ---------- WIP ----------
prompt_wip = "Jesteś ekspertem podatkowym specjalizującym się w wymianie informacji podatkowych z innymi państwami."
questions_wip = questions_main
PROMPT_WIP = {"prompt": prompt_wip, "questions": questions_wip}

# ---------- GLOBE ----------
prompt_globe = "Jesteś ekspertem podatkowym specjalizującym się w opodatkowaniu wyrównawczym jednostek składowych grup międzynarodowych i krajowych."
questions_globe = questions_main
PROMPT_GLOBE = {"prompt": prompt_globe, "questions": questions_globe}

# ---------- NIP ----------
prompt_nip = "Jesteś ekspertem podatkowym specjalizującym się w zasadach ewidencji i identyfikacji podatników i płatników (NIP)."
questions_nip = questions_main
PROMPT_NIP = {"prompt": prompt_nip, "questions": questions_nip}

# ---------- INNE ----------
prompt_inne = "Jesteś ekspertem podatkowym."
questions_inne = questions_main
PROMPT_INNE = {"prompt": prompt_inne, "questions": questions_inne}


PROMPTS_BY_AREA: Dict[Area, Dict[str, List[str] | str]] = {
    'vat': PROMPT_VAT,
    'pit': PROMPT_PIT,
    'cit': PROMPT_CIT,
    'pcc': PROMPT_PCC,
    'psd': PROMPT_PSD,
    'akcyza': PROMPT_AKCYZA,
    'op': PROMPT_OP,
    'gry': PROMPT_GRY,
    'malpki': PROMPT_MALPKI,
    'spw': PROMPT_SPW,
    'pt': PROMPT_PT,
    'pkop': PROMPT_PKOP,
    'spdet': PROMPT_SPDET,
    'fin': PROMPT_FIN,
    'cukier': PROMPT_CUKIER,
    'wip': PROMPT_WIP,
    'globe': PROMPT_GLOBE,
    'nip': PROMPT_NIP,
    'inne': PROMPT_INNE
}

def get_prompt(area: Area) -> str:
    """Zwraca wiodący prompt dla wskazanego obszaru."""
    return str(PROMPTS_BY_AREA[area]["prompt"])

def get_questions(area: Area) -> List[str]:
    """Zwraca listę pytań dla wskazanego obszaru."""
    return list(PROMPTS_BY_AREA[area]["questions"])  # kopia listy

def format_prompt_with_questions(area: Area) -> str:
    """
    Buduje końcowy tekst: prompt + wypunktowana lista pytań.
    Przydatne do przekazania modelowi jako jeden komunikat.
    """
    header = get_prompt(area).strip()
    items = get_questions(area)
    lines = [header, "", "Pytania:", *(f"- {q}" for q in items)]
    return "\n".join(lines)


if __name__ == "__main__":
    pass
    # ---------- Usage ----------
    # for type_tax in list(PROMPTS_BY_AREA.keys()):
    #     print(f"=== {type_tax} ===")
    #     print(format_prompt_with_questions(type_tax))

    # all_questions = {area: get_questions(area) for area in PROMPTS_BY_AREA.keys()}
    # print(all_questions)

