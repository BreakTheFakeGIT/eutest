from typing import Dict, List, Literal

Area = Literal['vat','pit','cit','pcc','psd','akcyza',
               'op','gry','malpki','spw','pt','pkop',
               'spdet','fin','cukier','wip','globe','nip','inne'
               ]

# ---------- MAIN ----------
questions_main: List[str] = [
    "W 5 punktach, wypisz 5 słów kluczowych.",
    "W 5 punktach, wypisz artykuły, przepisy wskazane w tekście.",
    "W 3 punktach, wypisz najważniejsze fragment tekstu."
]

# ---------- VAT ----------
prompt_vat: str = "Jesteś ekspertem podatkowym w zakresie podatku VAT."
questions_vat: List[str] = questions_main + [
    "W 2 zdaniach, opisz, czy wnioskodawca jest osobą fizyczną, jednostką samorządu terytorialnego (gmina, powiat, województwo), spółką, czy innym podmiotem (np. stowarzyszenie, spółdzielnia itp.)?",
    "W 2 zdaniach, opisz, zakres działalności gospodarczej prowadzonej przez wnioskodawcy?",
    "W 2 zdaniach, opisz, czy przysługiwało prawo do odliczenia podatku VAT od przedmiotowej transakcji?",
    "W 2 zdaniach, opisz, czy dostawa towaru lub świadczenia usługi, korzysta ze zwolnienia?",
    "W 2 zdaniach, opisz, jakiej sprzedaży towaru albo jakich świadczenia usług dotyczy wniosek?",
    "W 2 zdaniach, opisz, czy nieruchomość jest zabudowana bądź niezabudowana?",
    "W 2 zdaniach, opisz, czy wartość sprzedaży w roku podatkowym przekroczy kwotę 200000 zł?",
    "W 2 zdaniach, opisz, czy działalność jest opodatkowana podatkiem VAT, zwolniona, czy mieszana?",
    "W 2 zdaniach, opisz, czy wnioskodawca jest czynnym, zarejestrowanym podatnikiem VAT albo jako podatnik VAT UE?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy eksportu, importu, dokonuje transakcji wewnątrzwspólnotowych (WNT, WDT), korzysta w związku z WDT/WNT/eksportem ze stawki podatku VAT w wysokości 0%?",
    "W 2 zdaniach, opisz, na podstawie jakiej czynności prawnej wnioskodawca nabył towar?",
]
PROMPT_VAT: Dict[str, List[str] | str] = {"prompt": prompt_vat, "questions": questions_vat}

# ---------- PIT ----------
prompt_pit: str = "Jesteś ekspertem podatkowym w zakresie PIT."
questions_pit: List[str] = questions_main + [
    "W 2 zdaniach, opisz, czy wnioskodawca jest polskim rezydentem podatkowym oraz spełnia warunki rezydencji podatkowej tj.: miejsca zamieszkania, centrum interesów życiowych, długość pobytu?",
    "W 2 zdaniach, opisz, z jakich wnioskodawca korzysta z ulg i odliczeń podatkowych np.: na dzieci, rehabilitacyjna, termomodernizacyjna, ulga na powrót, na zabytki, mieszkaniowa, dla seniora, na innowacje?",
    "W 2 zdaniach, opisz, jaką wnioskodawca stosuje skalę podatkową (np.: podatek liniowy, ryczałt od przychodów ewidencjonowanych, karta podatkowa, podatek zryczałtowany)?",
    "W 2 zdaniach, opisz, jakie jest źródło przychodu wnioskodawcy (np.: z pracy, emerytury, renty, działalności wykonywanej osobiście, pozarolniczej działalności gospodarczej, działalności rolniczej lub działów specjalnych, najmu, kapitałów pieniężnych, zbycia nieruchomości/ruchomości, z zagranicy)?",
    "W 2 zdaniach, opisz, czy została wskazana forma opodatkowania dochodów/przychodów przez wnioskodawcę?",
    "W 2 zdaniach, opisz, określ formę prawną spółki (np.: zoo, akcyjna, jawna, komandytowa, cywilna)?",
    "W 2 zdaniach, opisz, czy wnioskodawca jest wspólnikiem spółki?",
    "W 2 zdaniach, opisz, czy wnioskodawca rozlicza podatek we własnym imieniu, czy działa jako płatnik (np.: pracodawca, zleceniodawca), występuje jako podatnik czy jako płatnik?",
]
PROMPT_PIT: Dict[str, List[str] | str] = {"prompt": prompt_pit, "questions": questions_pit}

# ---------- CIT ----------
prompt_cit: str = "Jesteś ekspertem podatkowym w zakresie CIT."
questions_cit: List[str] = questions_main + [
    "W 2 zdaniach, opisz, czy wnioskodawca korzysta z ulg podatkowych (np.: ulga badawczo-rozwojowa, ulga IP Box, zwolnienie dochodów z tytułu decyzji o wsparciu)?",
    "W 2 zdaniach, opisz, określ formę prawną spółki (np.: zoo, akcyjna, jawna, komandytowa, cywilna)?",
    "W 2 zdaniach, opisz, czy została wskazana forma opodatkowania dochodów/przychodów (np.: opodatkowanie na zasadach ogólnych, ryczałt od dochodów spółek, podatek u źródła)?",
    "W 2 zdaniach, opisz, czy wnioskodawca jest osobą fizyczną czy osobą prawną?",
    "W 2 zdaniach, opisz, czy wnioskodawca podlega opodatkowaniu podatkiem dochodowym od osób prawnych?",
    "W 2 zdaniach, opisz, czy wnioskodawca, jeśli posiada rezydencję podatkową w Polsce, podlega obowiązkowi podatkowemu od całości swoich dochodów bez względu na miejsce ich osiągania?\
    Jeśli nie posiada rezydencji podatkowej w Polsce, podlega obowiązkowi podatkowemu od dochodów, które osiąga na terytorium Polski?",
    "W 2 zdaniach, opisz, czy wnioskodawca występuje w charakterze podatnika czy płatnika?",
    "W 3 punktach, opisz, jaki jest zakres działalności prowadzonej przez wnioskodawcę?"
]
PROMPT_CIT: Dict[str, List[str] | str] = {"prompt": prompt_cit, "questions": questions_cit}

# ---------- PCC ----------
prompt_pcc = "Jesteś ekspertem podatkowym specjalizującym się w podatku od czynności cywilnoprawnych (PCC)."
questions_pcc = questions_main + [
    "W 2 zdaniach, opisz, jakiej umowy dotyczy wniosek (np.: umowy sprzedaży, umowy zamiany, umowy pożyczki, umowy darowizny, umowy dożywocia, umowy o dział spadku, umowy o zniesienie współwłasności, ustanowienia hipoteki, ustanowienia odpłatnego użytkowania, ustanowienia odpłatnej służebności, umowy depozytu nieprawidłowego, umowy spółki)?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy umowy spółki/zmiany umowy spółki? Jakiego rodzaju spółki?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy zasady stand still?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy wniesienia wkładu/aportu do spółki? Jakiego rodzaju spółki?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy przekształcenia spółki? Jaka spółka będzie przekształcana, a jaką będzie po przekształceniu?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy orzeczenia sądu lub ugody?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy sprzedaży prawa własności/użytkowania wieczystego nieruchomości?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy nabycia prawa własności lokalu mieszkalnego stanowiącego odrębną nieruchomość, prawa własności budynku mieszkalnego jednorodzinnego, spółdzielczego własnościowe prawa do lokalu dotyczącego lokalu mieszkalnego albo domu jednorodzinnego? Jakim tytułem?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy gospodarstwa rolnego?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy postępowania egzekucyjnego?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy cash poolingu?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy sprzedaży przedsiębiorstwa lub zorganizowanej części przedsiębiorstwa?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy podziału spółki przez wydzielenie lub wyodrębnienie?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy walut obcych?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy złota dewizowego/złota inwestycyjnego?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy spłat lub dopłat?",
]
PROMPT_PCC = {"prompt": prompt_pcc, "questions": questions_pcc}

# ---------- PSD ----------
prompt_psd = "Jesteś ekspertem podatkowym specjalizującym się w podatku od spadków i darowizn."
questions_psd = questions_main + [
    "W 2 zdaniach, opisz, czy wniosek dotyczy dziedziczenia, zapisu zwykłego, dalszego zapisu, zapisu windykacyjnego, polecenia testamentowego; darowizny, polecenia darczyńcy; zasiedzenia; nieodpłatnego zniesienia współwłasności; zachowku; nieodpłatnej renty, nieodpłatnego użytkowania, nieodpłatnej służebności?",
    "W 2 zdaniach, opisz, czy zostało wskazane, że był wydany akt poświadczenia dziedziczenia/ sądowe postanowienie o nabyciu spadku?",
    "W 2 zdaniach, opisz, czy wnioskodawca otrzyma/nabędzie od: małżonka (żona, mąż); zstępnego (córka, syn, wnuczka, wnuk, prawnuczka, prawnuk); wstępnego (matka, ojciec, babcia, dziadek, prababcia, pradziadek); pasierbicy, pasierba; siostry; brata; macochy, ojczyma?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy nabycia prawa do wkładu oszczędnościowego na podstawie dyspozycji wkładem na wypadek śmierci lub nabycia jednostek uczestnictwa na podstawie dyspozycji uczestnika funduszu inwestycyjnego otwartego albo specjalistycznego funduszu inwestycyjnego otwartego na wypadek jego śmierci?",
    "W 2 zdaniach, opisz, czy wnioskodawca ma obywatelstwo polskie lub kartę stałego pobytu w Polsce?",
    "W 2 zdaniach, opisz, czy rzecz lub prawo majątkowe znajduje się na terytorium RP?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy nabycia innego państwa? Jakiego?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy gospodarstwa rolnego?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy ulgi mieszkaniowej?",
]
PROMPT_PSD = {"prompt": prompt_psd, "questions": questions_psd}

# ---------- AKCYZA ----------
prompt_akcyza = "Jesteś ekspertem podatkowym specjalizującym się w podatku akcyzowym."
questions_akcyza = questions_main + [
    "W 2 zdaniach, opisz, jakiego rodzaju wyrobu dotyczy wniosek: wyrobu energetycznego, paliwa silnikowe, gaz ziemny, energii elektrycznej, napoju alkoholowego, wyrobu tytoniowego, suszu tytoniowego, płynu do papierosów elektronicznych, wyrobu nowatorskiego, urządzenia do waporyzacji, zestawu części do urządzeń do waporyzacji, saszetek nikotynowych, innych wyrobów nikotynowych albo samochodu osobowego?",
    "W 2 zdaniach, opisz, jaka jest klasyfikacja wyrobu akcyzowego albo samochodu osobowego w układzie odpowiadającym Nomenklaturze Scalonej (CN) albo rodzaj wyrobu akcyzowego?",
    "W 2 zdaniach, opisz, jakiej czynności dokonuje Wnioskodawca: zakup na terenie Polski, nabycie wewnątrzwspólnotowe, sprzedaż, zużycie, użycie, import, produkcja?",
    "W 2 zdaniach, opisz, czy przemieszczanie wyrobu odbywa się w lub poza procedurze zawieszenia poboru akcyzy?",
    "W 2 zdaniach, opisz, czy wyrób jest opodatkowany zerową stawką akcyzy, inną niż zerowa stawkę akcyzy czy podlega zwolnieniu od podatku?",
    "W 2 zdaniach, opisz, czy wnioskodawca jest producentem, importerem, eksporterem, dystrybutorem lub detalistą wyrobów akcyzowych?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy zwolnień lub obniżonych stawek akcyzy?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy procedur związanych z magazynowaniem, transportem, czy sprzedażą wyrobów akcyzowych?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy obowiązków sprawozdawczych lub ewidencyjnych związanych z podatkiem akcyzowym?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy sankcji lub kar związanych z naruszeniem przepisów akcyzowych?",
]
PROMPT_AKCYZA = {"prompt": prompt_akcyza, "questions": questions_akcyza}

# ---------- ORDYNACJA PODATKOWA ----------
prompt_op = "Jesteś ekspertem podatkowym specjalizującym się w ordynacji podatkowej."
questions_op = questions_main + [
    "W 2 zdaniach, opisz, czy wniosek dotyczy obowiązków podatkowych, praw podatników, procedur administracyjnych, kontroli podatkowej, odwołań i skarg, czy innych aspektów ordynacji podatkowej?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy e-deklaracji podatkowych? Jakiego rodzaju deklaracji (np. VAT, PIT, CIT, PCC)?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy terminów składania e-deklaracji, sposobu ich składania, czy problemów technicznych związanych z e-deklaracjami?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy ustawy o doręczeniach elektronicznych? Jakiego rodzaju doręczeń (np. decyzje administracyjne, wezwania, zawiadomienia)?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy procedur związanych z doręczeniami elektronicznymi, takich jak potwierdzenie odbioru, terminy doręczeń, czy problemy techniczne?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy sankcji lub kar związanych z naruszeniem przepisów ordynacji podatkowej lub ustawy o doręczeniach elektronicznych?",
]
PROMPT_OP = {"prompt": prompt_op, "questions": questions_op}

# ---------- GRY HAZARDOWE ----------
prompt_gry = "Jesteś ekspertem podatkowym specjalizującym się w grach hazardowych."
questions_gry = questions_main + [
    "W 2 zdaniach, opisz, czy wniosek dotyczy podatku od gier, opłaty za korzystanie z automatów do gier, czy innych aspektów podatkowych związanych z grami hazardowymi?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy organizatorów gier hazardowych, takich jak kasyna, salony gier, loterie, zakłady bukmacherskie?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy obowiązków podatkowych organizatorów gier hazardowych, takich jak rejestracja, ewidencja, raportowanie, czy płatności podatkowe?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy graczy, takich jak obowiązki podatkowe związane z wygranymi, zgłaszanie dochodów, czy ulgi podatkowe?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy sankcji lub kar związanych z naruszeniem przepisów podatkowych dotyczących gier hazardowych?",
]
PROMPT_GRY = {"prompt": prompt_gry, "questions": questions_gry}

# ---------- MALPKI ----------
prompt_malpki = "Jesteś ekspertem podatkowym specjalizującym się w wychowaniu w trzeźwości i przeciwdziałaniu alkoholizmowi."
questions_malpki = questions_main + [
    "W 2 zdaniach, opisz,  wnioskodawca sprzedaje napoje alkoholowe w opakowaniach jednostkowych o ilości nominalnej napoju nieprzekraczającej 300 ml?"
]
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
questions_cukier = questions_main + [
    "W 2 zdaniach, opisz, zwróć uwagę na skład napoju, czy dotyczy napoju z dodatkiem cukrów, soków owocowych/warzywnych, substancji słodzących, kofeiny lub tauryny?",
    "W 2 zdaniach, opisz, czy wnioskodawca jest podmiotem sprzedającym napoje do punktu sprzedaży detalicznej?",
    "W 2 zdaniach, opisz, czy wnioskodawca prowadzi sprzedaż detaliczną napojów?",
    "W 2 zdaniach, opisz, czy wnioskodawca jest zamawiającym napój u producenta?"
]
PROMPT_CUKIER = {"prompt": prompt_cukier, "questions": questions_cukier}

# ---------- WIP ----------
prompt_wip = "Jesteś ekspertem podatkowym specjalizującym się w wymianie informacji podatkowych z innymi państwami."
questions_wip = questions_main + [
    "W 2 zdaniach, opisz, czy wniosek dotyczy automatycznej wymiany o rachunkach raportowanych?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy automatycznej wymiany informacji o sprzedawcach?",
    "W 2 zdaniach, opisz, czy wniosek dotyczy automatycznej wymiany informacji podatkowych o jednostkach wchodzących w skład grupy podmiotów?"
]
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

