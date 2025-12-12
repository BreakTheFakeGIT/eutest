from typing import Tuple, List

# ---------- MAIN----------
questions_main = [
    "W punktach, wymień 10 słów kluczowych.",
    "W punktach, wymień artykuły, przepisy wksazane w tekście.",
    "W 2 punktach, podsumuj fragment tekstu."
    ]

# ---------- VAT ----------
prompt_vat = """Jesteś ekspertem podatkowym w zakresie podatku VAT. """
questions_vat = questions_main + [
"W 1 punkcie, wskaż, czy wnioskodawca prowadzi działalność gospodarczą?",
"W 1 punkcie, wskaż, czy wnioskodawca jest osobą fizyczną, jednostką samorządu terytorialnego (gmina, powiat, województwo), spółką, czy innym podmiotem (np. stowarzyszenie, spółdzielnia itp.)?",
"W 1 punkcie, wskaż, czy przysługiwało prawo do odliczenia podatku VAT od przedmiotowej transakcji?",
"W 1 punkcie, wskaż, czy dostawa towaru bądź świadczenie usługi korzysta ze zwolnienia?",
"W 1 punkcie, wskaż, czy wartość sprzedaży w roku podatkowym przekroczy kwotę 200.000 zł?",
"W 1 punkcie, wskaż, czy działalność jest opodatkowana podatkiem VAT, zwolniona, czy mieszana?",
"W 1 punkcie, wskaż, czy wnioskodawca korzysta w związku z WDT/WNT/eksportem ze stawki podatku VAT w wysokości 0%?"
"W 1 punkcie, wskaż, czy nieruchomość jest zabudowana bądź niezabudowana?",
"W 1 punkcie, wskaż, czy wnioskodawca jest czynnym, zarejestrowanym podatnikiem VAT?",
"W 1 punkcie, wskaż, czy wnioskodawca jest zarejestrowany jako podatnik VAT UE?",
"W 1 punkcie, wskaż, czy wniosek dotyczy eksportu czy wniosek dotyczy importu?",
"W 1 punkcie, wskaż, czy wnioskodawca dokonuje transakcji wewnątrzwspólnotowych (WNT, WDT)?",
"W 3 punktach, wskaż, jaki jest zakres działalności gospodarczej prowadzonej przez wnioskodawcę?",
"W 3 punktach, wskaż, czy wniosek dotyczy sprzedaży towaru (jakiego)?",
"W 3 punktach, wskaż, czy wniosek dotyczy świadczenia usług (jakich)?",
"W 3 punktach, wskaż, na podstawie jakiej czynności prawnej wnioskodawca nabył towar?",
]


PROMPT_VAT = {'prompt': prompt_vat , 'questions': questions_vat}

# ---------- PIT ----------
prompt_pit = """Jesteś ekspertem podatkowym w zakresie PIT. """
questions_pit = questions_main + [
"W 1 punkcie, wskaż, czy wnioskodawca jest polskim rezydentem podatkowym?",
"W 1 punkcie, wskaż, czy wnioskodawca spełnia warunki rezydencji podatkowej tj.: miejsca zamieszkania, centrum interesów życiowych, długość pobytu?",
"W 1 punkcie, wskaż, czy wnioskodawca korzysta z ulg i odliczeń podatkowych?",
"W 1 punkcie, wskaż, jakie zastosowano ulgi (np.: na dzieci, rehabilitacyjna, termomodernizacyjna, ulga na powrót, na zabytki, mieszkaniowa, dla seniora, na innowacje)?",
"W 1 punkcie, wskaż, jaką wnioskodawca stosuje skalę podatkową (np.: podatek liniowy, ryczałt od przychodów ewidencjonowanych, kartę podatkową, podatek zryczałtowany)?",
"W 1 punkcie, wskaż, jakie jest źródło przychodu wnioskodawcy?",
"W 1 punkcie, wskaż, skąd pochodzi przychód pochodzi (np.: z pracy, emerytury, renty, działalności wykonywanej osobiście, pozarolniczej działalności gospodarczej, działalności rolniczej lub działów specjalnych, najmu, kapitałów pieniężnych, zbycia nieruchomości/ruchomości, z zagranicy)?",
"W 1 punkcie, wskaż, czy została wskazana forma opodatkowania dochodów/przychodów przez wnioskodawcę?",
"W 1 punkcie, wskaż, określ formę prawną spółki (np.: jawna, komandytowa, cywilna)?",
"W 1 punkcie, wskaż, czy wnioskodawca jest wspólnikiem spółki?",
"W 1 punkcie, wskaż, czy wnioskodawca rozlicza podatek we własnym imieniu, czy działa jako płatnik (np.: pracodawca, zleceniodawca)?",
"W 1 punkcie, wskaż, czy wnioskodawca występuje jako podatnik czy jako płatnik?",
]


PROMPT_PIT  = {'prompt': prompt_pit, 'questions': questions_pit}


# ---------- CIT ----------
prompt_cit = """Jesteś ekspertem podatkowym w zakresie CIT. """ 
questions_cit = questions_main + [
    "W 1 punkcie, wskaż, czy wnioskodawca korzysta z ulg podatkowych (np.: ulga badawczo-rozwojowa, ulga IP Box, zwolnienie dochodów z tytułu decyzji o wsparciu)?",
    "W 1 punkcie, wskaż, jaka jest forma prawna wnioskodawcy?",
    "W 1 punkcie, wskaż, czy została wskazana forma opodatkowania dochodów / przychodów (np.: opodatkowanie na zasadach ogólnych, ryczałt od dochodów spółek, podatek u źródła)?",
    "W 1 punkcie, wskaż, czy wnioskodawca podlega opodatkowaniu podatkiem dochodowym od osób prawnych?",
    "W 1 punkcie, wskaż, czy wnioskodawca jest osobą fizyczną czy osobą prawną?",
    "W 1 punkcie, wskaż, czy wnioskodawca, jeśli posiada rezydencję podatkową w Polsce to podlega obowiązkowi podatkowemu od całości swoich dochodów bez względu na miejsce ich osiągania?",
    "W 1 punkcie, wskaż, czy wnioskodawca, jesli nie posiada rezydencję podatkową w Polsce to podlega obowiązkowi podatkowemu od dochodów, które osiąga na terytorium Polski?",
    "W 1 punkcie, wskaż, czy wnioskodawca występuje w charakterze podatnika czy płatnika?",
    "W 3 punktach, wskaż, jaki jest zakres działalności prowadzonej przez wnioskodawcę?"
   ]

PROMPT_CIT  = {'prompt': prompt_cit, 'questions': questions_cit}

# ---------- PCC ----------
prompt_pcc = """Jesteś ekspertem podatkowym specjalizującym się w podatku od czynności cywilnoprawnych (PCC). """ 
questions_pcc = questions_main + [
"W 1 punkcie, wskaż, jakiej umowy dotyczy wniosek (np.: umowy sprzedaży, umowy zamiany, umowy pożyczki, umowy darowizny, umowy dożywocia, umowy o dział spadku, umowy o zniesienie współwłasności, ustanowienia hipoteki, ustanowienia odpłatnego użytkowania, ustanowienia odpłatnej służebności, umowy depozytu nieprawidłowego, umowy spółki)?",
"W 1 punkcie, wskaż, czy wniosek dotyczy umowy spółki/zmiany umowy spółki? Jakiego rodzaju spółki?",
"Czy wniosek dotyczy zasady stand still?",
"Czy wniosek dotyczy wniesienia wkładu/aportu do spółki? Jakiego rodzaju spółki?",
"Czy wniosek dotyczy przekształcenia spółki? Jaka spółka będzie przekształcana, a jaką będzie po przekształceniu?",
"Czy wniosek dotyczy orzeczenia sądu lub ugody?",
"Czy wniosek dotyczy sprzedaży prawa własności/użytkowania wieczystego nieruchomości?",
"Czy wniosek dotyczy nabycia prawa własności lokalu mieszkalnego stanowiącego odrębną nieruchomość, prawa własności budynku mieszkalnego jednorodzinnego, spółdzielczego własnościowe prawa do lokalu dotyczącego lokalu mieszkalnego albo domu jednorodzinnego? Jakim tytułem?",
"Czy wniosek dotyczy gospodarstwa rolnego?",
"Czy wniosek dotyczy postępowania egzekucyjnego?",
"Czy wniosek dotyczy cash poolingu?",
"Czy wniosek dotyczy sprzedaży przedsiębiorstwa lub zorganizowanej części przedsiębiorstwa?",
"Czy wniosek dotyczy podziału spółki przez wydzielenie lub wyodrębnienie?",
"Czy wniosek dotyczy walut obcych?",
"Czy wniosek dotyczy złota dewizowego/złota inwestycyjnego?",
"Czy wniosek dotyczy spłat lub dopłat?",
    ]

PROMPT_PCC  = {'prompt': prompt_pcc, 'questions': questions_pcc}


# ---------- PSD ----------
prompt_psd = """Jesteś ekspertem podatkowym specjalizującym się w podatku od spadków i darowizn. """ 
questions_psd = questions_main + [
    "Czy treść dotyczy dziedziczenia, zapisu zwykłego, dalszego zapisu, zapisu windykacyjnego, polecenia testamentowego; darowizny, polecenia darczyńcy; zasiedzenia; nieodpłatnego zniesienia współwłasności; zachowku; nieodpłatnej renty, nieodpłatnego użytkowania, nieodpłatnej służebności?",
    "Czy treść dotyczy nabycia prawa do wkładu oszczędnościowego na podstawie dyspozycji wkładem na wypadek śmierci lub nabycia jednostek uczestnictwa na podstawie dyspozycji uczestnika funduszu inwestycyjnego otwartego albo specjalistycznego funduszu inwestycyjnego otwartego na wypadek jego śmierci?",
    "Czy wnioskodawca ma obywatelstwo polskie lub kartę stałego pobytu w Polsce?",
    "Czy rzecz lub prawo majątkowe znajduje się na terytorium RP?",
    "Czy treść dotyczy nabycia innego państwa? Jakiego?",
    "Czy wnioskodawca otrzyma/nabędzie od: małżonka (żona, mąż); zstępnego (córka, syn, wnuczka, wnuk, prawnuczka, prawnuk); wstępnego (matka, ojciec, babcia, dziadek, prababcia, pradziadek); pasierbicy, pasierba; siostry; brata; macochy, ojczyma?",
    "Czy treść dotyczy gospodarstwa rolnego?",
    "Czy treść dotyczy ulgi mieszkaniowej?",
    "Czy zostało wskazane, że był wydany akt poświadczenia dziedziczenia/ sądowe postanowienie o nabyciu spadku?"
    ]


PROMPT_PSD  = {'prompt': prompt_psd, 'questions': questions_psd}

# ---------- AKCYZA ----------
prompt_akcyza = """Jesteś ekspertem podatkowym specjalizującym się w podatku akcyzowym. """ 
questions_akcyza = questions_main + [
    "Jakiego rodzaju wyrobu dotyczy treść: wyrobu energetycznego,paliwa silnikowe, gaz ziemny energii elektrycznej, napoju alkoholowego, wyrobu tytoniowego, suszu tytoniowego, płynu do papierosów elektronicznych, wyrobu nowatorskiego, urządzenia do waporyzacji, zestawu części do urządzeń do waporyzacji, saszetek nikotynowych, innych wyrobów nikotynowych albo samochodu osobowego?",
    "Jaka jest klasyfikacja wyrobu akcyzowego albo samochodu osobowego w układzie odpowiadającym Nomenklaturze Scalonej (CN) albo rodzaj wyrobu akcyzowego?",
    "Jakiej czynności dokonuje Wnioskodawca: zakup na terenie Polski, nabycie wewnątrzwspólnotowe, sprzedaż, zużycie, użycie, import, produkcja?",
    "Czy przemieszczanie wyrobu odbywa się w procedurze zawieszenia poboru akcyzy?",
    "Czy też przemieszczanie wyrobu odbywa się poza procedurą zawieszenia poboru akcyzy?",
    "Czy wyrób jest opodatkowany zerową stawką akcyzy, inną niż zerowa stawka akcyzy czy podlega zwolnieniu od podatku?",
    "Czy wnioskodawca jest producentem, importerem, eksporterem, dystrybutorem lub detalistą wyrobów akcyzowych?",
    "Czy treść dotyczy zwolnień lub obniżonych stawek akcyzy?",
    "Czy treść dotyczy procedur związanych z magazynowaniem, transportem, czy sprzedażą wyrobów akcyzowych?",
    "Czy treść dotyczy obowiązków sprawozdawczych lub ewidencyjnych związanych z podatkiem akcyzowym?",
    "Czy treść dotyczy sankcji lub kar związanych z naruszeniem przepisów akcyzowych?"
]

PROMPT_AKCYZA  = {'prompt': prompt_akcyza, 'questions': questions_akcyza}


# ---------- ORDYNACJA PODATKOW ----------
prompt_op = """Jesteś ekspertem podatkowym specjalizującym się w ordynacji podatkowej.""" 
questions_op = questions_main + [
    "Czy treść dotyczy obowiązków podatkowych, praw podatników, procedur administracyjnych, kontroli podatkowej, odwołań i skarg, czy innych aspektów ordynacji podatkowej?",
    "Czy treść dotyczy e-deklaracji podatkowych? Jakiego rodzaju deklaracji (np. VAT, PIT, CIT, PCC)?",
    "Czy treść dotyczy terminów składania e-deklaracji, sposobu ich składania, czy problemów technicznych związanych z e-deklaracjami?",
    "Czy treść dotyczy ustawy o doręczeniach elektronicznych? Jakiego rodzaju doręczeń (np. decyzje administracyjne, wezwania, zawiadomienia)?",
    "Czy treść dotyczy procedur związanych z doręczeniami elektronicznymi, takich jak potwierdzenie odbioru, terminy doręczeń, czy problemy techniczne?",
    "Czy treść dotyczy sankcji lub kar związanych z naruszeniem przepisów ordynacji podatkowej lub ustawy o doręczeniach elektronicznych?"
    ]

PROMPT_OP  = {'prompt': prompt_op, 'questions': questions_op}


# ---------- GRY HAZARDOWE ----------
prompt_gry = """Jesteś ekspertem podatkowym specjalizującym się w grach hazardowych.""" 
questions_gry = questions_main + [
    "Czy treść dotyczy podatku od gier, opłaty za korzystanie z automatów do gier, czy innych aspektów podatkowych związanych z grami hazardowymi?",
    "Czy treść dotyczy organizatorów gier hazardowych, takich jak kasyna, salony gier, loterie, zakłady bukmacherskie?",
    "Czy treść dotyczy obowiązków podatkowych organizatorów gier hazardowych, takich jak rejestracja, ewidencja, raportowanie, czy płatności podatkowe?",
    "Czy treść dotyczy graczy, takich jak obowiązki podatkowe związane z wygranymi, zgłaszanie dochodów, czy ulgi podatkowe?",
    "Czy treść dotyczy sankcji lub kar związanych z naruszeniem przepisów podatkowych dotyczących gier hazardowych?"
]

PROMPT_GRY  = {'prompt': prompt_gry, 'questions': questions_gry}

# ---------- MALPKI  ----------
prompt_malpki = """Jesteś ekspertem podatkowym specjalizującym się w wychowaniu w trzeźwości i przeciwdziałaniu alkoholizmowi.""" 
questions_malpki  = questions_main + [
    "Czy wnioskodawca sprzedaje napoje alkoholowe w opakowaniach jednostkowych o ilości nominalnej napoju nieprzekraczającej 300 ml?"
    ]
PROMPT_MALPKI  = {'prompt': prompt_malpki ,'questions': questions_malpki}


# ---------- WĘGLOWODORY ----------
prompt_spw = """Jesteś ekspertem podatkowym specjalizującym się w podatku węglowodorowym.""" 
questions_spw = questions_main

PROMPT_SPW = {'prompt': prompt_spw ,'questions': questions_spw}

# ---------- TONAŻ ----------
prompt_pt = """Jesteś ekspertem podatkowym specjalizującym się w podateku tonażowym.""" 
questions_pt = questions_main

PROMPT_PT = {'prompt': prompt_pt ,'questions': questions_pt}

# ---------- KOPALINY ----------
prompt_pkop = """Jesteś ekspertem podatkowym specjalizującym się w podatku od wydobycia niektórych kopalin.""" 
questions_pkop = questions_main

PROMPT_PKOP = {'prompt': prompt_pkop ,'questions': questions_pkop}

# ---------- SPRZEDAZ ----------
prompt_spdet = """Jesteś ekspertem podatkowym specjalizującym się w podateku od sprzedaży detalicznej.""" 
questions_spdet  = questions_main

PROMPT_SPDET = {'prompt': prompt_spdet ,'questions': questions_spdet}

# ---------- FINANSE ----------
prompt_fin = """Jesteś ekspertem podatkowym specjalizującym się w podatku od niektórych instytucji finansowych.""" 
questions_fin = questions_main

PROMPT_FIN = {'prompt': prompt_fin ,'questions': questions_fin}

# ---------- CUKIER ----------
prompt_cukier = """Jesteś ekspertem podatkowym specjalizującym się w zdrowiu publicznym.""" 
questions_cukier = questions_main + [
    "Zwróć uwagę na skład napoju, czy dotyczy napoju z dodatkiem cukrów, soków owocowych/warzywnych, substancji słodzących, kofeiny lub tauryny?",
    "Czy wnioskodawca jest podmiotem sprzedającym napoje do punktu sprzedaży detalicznej?",
    "Czy wnioskodawca prowadzi sprzedaż detaliczną napojów?",
    "Czy wnioskodawca jest zamawiającym napój u producenta?"
    ]

PROMPT_CUKIER = {'prompt': prompt_cukier ,'questions': questions_cukier}

# ---------- WIP ----------
prompt_wip = """Jesteś ekspertem podatkowym specjalizującym się w wymianie informacji podatkowych z innymi państwami.""" 
questions_wip = questions_main + [
    "Czy treść dotyczy automatycznej wymiany o rachunkach raportowanych?",
    "Czy treść dotyczy automatycznej wymiany informacji o sprzedawcach?",
    "Czy treść dotyczy automatycznej wymiana informacji podatkowych o jednostkach wchodzących w skład grupy podmiotów?"
    ]

PROMPT_WIP = {'prompt': prompt_wip ,'questions': questions_wip}

# ---------- GLOBE ----------
prompt_globe = """Jesteś ekspertem podatkowym specjalizującym się w opodatkowaniu wyrównawczym jednostek składowych grup międzynarodowych i krajowych.""" 
questions_globe = questions_main

PROMPT_GLOBE = {'prompt': prompt_globe ,'questions': questions_globe}

# ---------- NIP ----------
prompt_nip = """Jesteś ekspertem podatkowym specjalizującym się w zasadach ewidencji i identyfikacji podatników i płatników (NIP).""" 
questions_nip = questions_main

PROMPT_NIP= {'prompt': prompt_nip ,'questions': questions_nip}

# ---------- INNE ----------
prompt_inne = """Jesteś ekspertem podatkowym.""" 
questions_inne = questions_main

PROMPT_INNE= {'prompt': prompt_inne ,'questions': questions_inne}



# ---------- TAX_PROMPTS ----------
TAX_PROMPTS = {'vat': PROMPT_VAT,
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

def tax_prompts(tax_type: str) -> Tuple[str,List[str]]:
    prompt_template = TAX_PROMPTS.get(tax_type, TAX_PROMPTS["inne"])
    questions = prompt_template.get('questions', [])
    prompt = prompt_template.get('prompt', [])
    return prompt, questions


if __name__ == "__main__":
    pass

    # prompt, questions = tax_prompts(tax_type='vat', user_text='abcd')
    # print(prompt)
    # print(questions)