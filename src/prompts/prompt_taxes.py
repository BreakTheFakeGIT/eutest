from typing import Tuple, List

# ---------- MAIN----------
prompt_main = """\n\nOdpowiedz na pytania do wniosku krótko i zrozumiale. Jeśli nie znasz odpowiedzi lub odpowiedź jest nie/negatywna napisz krótko: "Brak informacji".
Wnioskodawca wprowadził wniosek: {user_text}.\n\n"""
questions_main = [
    "Jaki rodzaj podatku wskazano we wniosku?",
    "Jakie wskazano artykuły, przepisy prawa podatkowego?",
    "Jakie jest 10 kluczowych słów we wniosku?",
    "Napisz w 2 zdaniach o co pyta wnioskodawca."
    ]


# ---------- VAT ----------
prompt_vat = """Jesteś ekspertem w zakresie podatku VAT, obejmującym odliczenia i zwroty, zwolnienia, obniżone stawki, zasady fakturowania i JPK, wspólny system VAT 
oraz szczególne przypadki dotyczące pojazdów, importu i zwrotów dla sił zbrojnych. """ + prompt_main
questions_vat = questions_main + [
    "Czy wnioskodawca jest osobą fizyczną, jednostką samorządu terytorialnego (gmina, powiat, województwo), spółką, czy innym podmiotem (np. stowarzyszenie, spółdzielnia itp.)?",
    "Czy wnioskodawca jest czynnym, zarejestrowanym podatnikiem VAT?",
    "Czy wnioskodawca jest zarejestrowany jako podatnik VAT UE?",
    "Czy wnioskodawca prowadzi działalność gospodarczą?",
    "Czy działalność jest opodatkowana podatkiem VAT, zwolniona, czy mieszana?",
    "Jaki jest zakres działalności gospodarczej prowadzonej przez wnioskodawcę?",
    "Czy wniosek dotyczy sprzedaży towaru (jakiego)?",
    "Czy wniosek dotyczy świadczenia usług (jakich)?",
    "Na podstawie jakiej czynności prawnej wnioskodawca nabył towar?",
    "Czy wniosek dotyczy eksportu lub importu?",
    "Czy wnioskodawca dokonuje transakcji wewnątrzwspólnotowych (WNT, WDT)?",
    "Czy przysługiwało prawo do odliczenia podatku VAT od przedmiotowej transakcji?",
    "Czy nieruchomość jest zabudowana bądź niezabudowana?",
    "Czy wartość sprzedaży w roku podatkowym przekroczy kwotę 200.000 zł?",
    "Czy dostawa towaru bądź świadczenie usługi korzysta ze zwolnienia?",
    "Czy wnioskodawca korzysta w związku z WDT/WNT/eksportem ze stawki podatku VAT w wysokości 0%?"
    ]

PROMPT_VAT = {'prompt': prompt_vat , 'questions': questions_vat}

# ---------- PIT ----------
prompt_pit = """Jesteś ekspertem w zakresie PIT, ewidencji przychodów i środków trwałych, zwrotów wydatków mieszkaniowych oraz programów wsparcia dla młodych przy zakupie pierwszego mieszkania.""" + prompt_main
questions_pit = questions_main + [
    "Czy wnioskodawca jest polskim rezydentem podatkowym?",
    "Czy wnioskodawca spełnia warunki rezydencji podatkowej zgodnie z art. 3 ustawy PIT (miejsce zamieszkania, centrum interesów życiowych, długość pobytu)?",
    "Czy wnioskodawca występuje jako podatnik czy jako płatnik?",
    "Czy wnioskodawca rozlicza podatek we własnym imieniu, czy działa jako płatnik (np. pracodawca, zleceniodawca)?",
    "Jakie jest źródło przychodu wnioskodawcy?",
    "Czy przychód pochodzi z pracy, emerytury, renty, działalności wykonywanej osobiście, pozarolniczej działalności gospodarczej, działalności rolniczej lub działów specjalnych, najmu, kapitałów pieniężnych, zbycia nieruchomości/ruchomości, z zagranicy?",
    "Czy wnioskodawca jest wspólnikiem spółki?",
    "Określ formę prawną spółki (np. jawna, komandytowa, cywilna) i konsekwencje podatkowe wynikające z uczestnictwa?",
    "Czy została wskazana forma opodatkowania dochodów/przychodów przez wnioskodawcę?",
    "Czy wnioskodawca stosuje skalę podatkową, podatek liniowy, ryczałt od przychodów ewidencjonowanych, kartę podatkową, podatek zryczałtowany?",
    "Czy wnioskodawca korzysta z ulg i odliczeń podatkowych? – Zidentyfikuj zastosowane ulgi (np. na dzieci, rehabilitacyjna, termomodernizacyjna, ulga na powrót, na zabytki, mieszkaniowa, dla seniora, na innowacje) oraz ich wpływ na zobowiązanie podatkowe?"
    ]

PROMPT_PIT  = {'prompt': prompt_pit, 'questions': questions_pit}


# ---------- CIT ----------
prompt_cit = """Jesteś ekspertem w zakresie CIT, cen transferowych, eliminacji podwójnego opodatkowania, dokumentacji podatkowej, gospodarki finansowej przedsiębiorstw państwowych oraz wpłat z zysków spółek Skarbu Państwa.""" + prompt_main
questions_cit = questions_main + [
    "Czy wnioskodawca podlega opodatkowaniu podatkiem dochodowym od osób prawnych?",
    "Czy wnioskodawca jest osobą fizyczną czy osobą prawną?",
    "Jaka jest forma prawna wnioskodawcy?",
    "Czy wnioskodawca podlega w Polsce obowiązkowi podatkowemu od całości swoich dochodów bez względu na miejsce ich osiągania (posiada rezydencję podatkową w Polsce), czy tylko od dochodów, które osiąga na terytorium Polski (nie posiada rezydencję podatkową w Polsce)?",
    "Czy wnioskodawca występuje w charakterze podatnika czy płatnika?",
    "Czy została wskazana forma opodatkowania dochodów / przychodów? (np. opodatkowanie na zasadach ogólnych, ryczałt od dochodów spółek, podatek u źródła)",
    "Czy wnioskodawca korzysta z ulg podatkowych (np. ulga badawczo-rozwojowa, ulga IP Box, zwolnienie dochodów z tytułu decyzji o wsparciu, itd.)?",
    "Jaki jest zakres działalności prowadzonej przez wnioskodawcę?"
   ]

PROMPT_CIT  = {'prompt': prompt_cit, 'questions': questions_cit}

# ---------- PCC ----------
prompt_pcc = """Jesteś ekspertem podatkowym specjalizującym się w podatku od czynności cywilnoprawnych (PCC).""" + prompt_main
questions_pcc = questions_main + [
    "Czy wniosek dotyczy umowy sprzedaży, umowy zamiany, umowy pożyczki, umowy darowizny, umowy dożywocia, umowy o dział spadku, umowy o zniesienie współwłasności, ustanowienia hipoteki, ustanowienia odpłatnego użytkowania, ustanowienia odpłatnej służebności, umowy depozytu nieprawidłowego, umowy spółki / lub zmiany takich umów?",
    "Czy wniosek dotyczy umowy spółki/zmiany umowy spółki? Jakiego rodzaju spółki?",
    "Czy wniosek dotyczy zasady stand still?",
    "Czy wniosek dotyczy wniesienia wkładu/aportu do spółki? Jakiego rodzaju spółki?",
    "Czy wniosek dotyczy przekształcenia spółki? Jaka spółka będzie przekształcana, a jaką będzie po przekształceniu?",
    "Czy wniosek dotyczy orzeczenia sądu lub ugody?",
    "Czy wniosek dotyczy sprzedaży prawa własności lub użytkowania wieczystego nieruchomości?",
    "Czy wniosek dotyczy nabycia prawa własności lokalu mieszkalnego stanowiącego odrębną nieruchomość, prawa własności budynku mieszkalnego jednorodzinnego, spółdzielczego własnościowe prawa do lokalu dotyczącego lokalu mieszkalnego albo domu jednorodzinnego? Jakim tytułem?",
    "Czy wniosek dotyczy gospodarstwa rolnego?",
    "Czy wniosek dotyczy postępowania egzekucyjnego?",
    "Czy wniosek dotyczy cash poolingu?",
    "Czy wniosek dotyczy sprzedaży przedsiębiorstwa lub zorganizowanej części przedsiębiorstwa?",
    "Czy wniosek dotyczy podziału spółki przez wydzielenie lub wyodrębnienie?",
    "Czy wniosek dotyczy walut obcych?",
    "Czy wniosek dotyczy złota dewizowego/złota inwestycyjnego?",
    "Czy wniosek dotyczy spłat lub dopłat?"
    ]

PROMPT_PCC  = {'prompt': prompt_pcc, 'questions': questions_pcc}


# ---------- PSD ----------
prompt_psd = """Jesteś ekspertem podatkowym specjalizującym się w podatku od spadków i darowizn.""" + prompt_main
questions_psd = questions_main + [
    "Czy wniosek dotyczy dziedziczenia, zapisu zwykłego, dalszego zapisu, zapisu windykacyjnego, polecenia testamentowego; darowizny, polecenia darczyńcy; zasiedzenia; nieodpłatnego zniesienia współwłasności; zachowku; nieodpłatnej renty, nieodpłatnego użytkowania, nieodpłatnej służebności?",
    "Czy wniosek dotyczy nabycia prawa do wkładu oszczędnościowego na podstawie dyspozycji wkładem na wypadek śmierci lub nabycia jednostek uczestnictwa na podstawie dyspozycji uczestnika funduszu inwestycyjnego otwartego albo specjalistycznego funduszu inwestycyjnego otwartego na wypadek jego śmierci?",
    "Czy wnioskodawca ma obywatelstwo polskie lub kartę stałego pobytu w Polsce?",
    "Czy rzecz lub prawo majątkowe znajduje się na terytorium RP?",
    "Czy wniosek dotyczy nabycia innego państwa? Jakiego?",
    "Czy wnioskodawca otrzyma/nabędzie od: małżonka (żona, mąż); zstępnego (córka, syn, wnuczka, wnuk, prawnuczka, prawnuk); wstępnego (matka, ojciec, babcia, dziadek, prababcia, pradziadek); pasierbicy, pasierba; siostry; brata; macochy, ojczyma?",
    "Czy wniosek dotyczy gospodarstwa rolnego?",
    "Czy wniosek dotyczy ulgi mieszkaniowej?",
    "Czy zostało wskazane, że był wydany akt poświadczenia dziedziczenia/ sądowe postanowienie o nabyciu spadku?"
    ]


PROMPT_PSD  = {'prompt': prompt_psd, 'questions': questions_psd}

# ---------- AKCYZA ----------
prompt_akcyza = """Jesteś ekspertem podatkowym specjalizującym się w podatku akcyzowym.""" + prompt_main
questions_akcyza = questions_main + [
    "Jakiego rodzaju wyrobu dotyczy wniosek: wyrobu energetycznego,paliwa silnikowe, gaz ziemny energii elektrycznej, napoju alkoholowego, wyrobu tytoniowego, suszu tytoniowego, płynu do papierosów elektronicznych, wyrobu nowatorskiego, urządzenia do waporyzacji, zestawu części do urządzeń do waporyzacji, saszetek nikotynowych, innych wyrobów nikotynowych albo samochodu osobowego?",
    "Jaka jest klasyfikacja wyrobu akcyzowego albo samochodu osobowego w układzie odpowiadającym Nomenklaturze Scalonej (CN) albo rodzaj wyrobu akcyzowego?",
    "Jakiej czynności dokonuje Wnioskodawca: zakup na terenie Polski, nabycie wewnątrzwspólnotowe, sprzedaż, zużycie, użycie, import, produkcja?",
    "Czy przemieszczanie wyrobu odbywa się w procedurze zawieszenia poboru akcyzy?",
    "Czy też przemieszczanie wyrobu odbywa się poza procedurą zawieszenia poboru akcyzy?",
    "Czy wyrób jest opodatkowany zerową stawką akcyzy, inną niż zerowa stawka akcyzy czy podlega zwolnieniu od podatku?",
    "Czy wnioskodawca jest producentem, importerem, eksporterem, dystrybutorem lub detalistą wyrobów akcyzowych?",
    "Czy wniosek dotyczy zwolnień lub obniżonych stawek akcyzy?",
    "Czy wniosek dotyczy procedur związanych z magazynowaniem, transportem, czy sprzedażą wyrobów akcyzowych?",
    "Czy wniosek dotyczy obowiązków sprawozdawczych lub ewidencyjnych związanych z podatkiem akcyzowym?",
    "Czy wniosek dotyczy sankcji lub kar związanych z naruszeniem przepisów akcyzowych?"
]

PROMPT_AKCYZA  = {'prompt': prompt_akcyza, 'questions': questions_akcyza}


# ---------- ORDYNACJA PODATKOW ----------
prompt_op = """Jesteś ekspertem podatkowym specjalizującym się w ordynacji podatkowej, e-deklaracjach oraz ustawy o doręczeniach elektronicznych.""" + prompt_main
questions_op = questions_main + [
    "Czy wniosek dotyczy obowiązków podatkowych, praw podatników, procedur administracyjnych, kontroli podatkowej, odwołań i skarg, czy innych aspektów ordynacji podatkowej?",
    "Czy wniosek dotyczy e-deklaracji podatkowych? Jakiego rodzaju deklaracji (np. VAT, PIT, CIT, PCC)?",
    "Czy wniosek dotyczy terminów składania e-deklaracji, sposobu ich składania, czy problemów technicznych związanych z e-deklaracjami?",
    "Czy wniosek dotyczy ustawy o doręczeniach elektronicznych? Jakiego rodzaju doręczeń (np. decyzje administracyjne, wezwania, zawiadomienia)?",
    "Czy wniosek dotyczy procedur związanych z doręczeniami elektronicznymi, takich jak potwierdzenie odbioru, terminy doręczeń, czy problemy techniczne?",
    "Czy wniosek dotyczy sankcji lub kar związanych z naruszeniem przepisów ordynacji podatkowej lub ustawy o doręczeniach elektronicznych?"
    ]

PROMPT_OP  = {'prompt': prompt_op, 'questions': questions_op}


# ---------- GRY HAZARDOWE ----------
prompt_gry = """Jesteś ekspertem podatkowym specjalizującym się w grach hazardowych.""" + prompt_main
questions_gry = questions_main + [
    "Czy wniosek dotyczy podatku od gier, opłaty za korzystanie z automatów do gier, czy innych aspektów podatkowych związanych z grami hazardowymi?",
    "Czy wniosek dotyczy organizatorów gier hazardowych, takich jak kasyna, salony gier, loterie, zakłady bukmacherskie?",
    "Czy wniosek dotyczy obowiązków podatkowych organizatorów gier hazardowych, takich jak rejestracja, ewidencja, raportowanie, czy płatności podatkowe?",
    "Czy wniosek dotyczy graczy, takich jak obowiązki podatkowe związane z wygranymi, zgłaszanie dochodów, czy ulgi podatkowe?",
    "Czy wniosek dotyczy sankcji lub kar związanych z naruszeniem przepisów podatkowych dotyczących gier hazardowych?"
]

PROMPT_GRY  = {'prompt': prompt_gry, 'questions': questions_gry}

# ---------- MALPKI  ----------
prompt_malpki = """Jesteś ekspertem podatkowym specjalizującym się w wychowaniu w trzeźwości i przeciwdziałaniu alkoholizmowi.""" + prompt_main
questions_malpki  = questions_main + [
    "Czy wnioskodawca sprzedaje napoje alkoholowe w opakowaniach jednostkowych o ilości nominalnej napoju nieprzekraczającej 300 ml?"
    ]
PROMPT_MALPKI  = {'prompt': prompt_malpki ,'questions': questions_malpki}


# ---------- WĘGLOWODORY ----------
prompt_spw = """Jesteś ekspertem podatkowym specjalizującym się w podatku węglowodorowym.""" + prompt_main
questions_spw = questions_main

PROMPT_SPW = {'prompt': prompt_spw ,'questions': questions_spw}

# ---------- TONAŻ ----------
prompt_pt = """Jesteś ekspertem podatkowym specjalizującym się w podateku tonażowym.""" + prompt_main
questions_pt = questions_main

PROMPT_PT = {'prompt': prompt_pt ,'questions': questions_pt}

# ---------- KOPALINY ----------
prompt_pkop = """Jesteś ekspertem podatkowym specjalizującym się w podatku od wydobycia niektórych kopalin.""" + prompt_main
questions_pkop = questions_main

PROMPT_PKOP = {'prompt': prompt_pkop ,'questions': questions_pkop}

# ---------- SPRZEDAZ ----------
prompt_spdet = """Jesteś ekspertem podatkowym specjalizującym się w podateku od sprzedaży detalicznej.""" + prompt_main
questions_spdet  = questions_main

PROMPT_SPDET = {'prompt': prompt_spdet ,'questions': questions_spdet}

# ---------- FINANSE ----------
prompt_fin = """Jesteś ekspertem podatkowym specjalizującym się w podatku od niektórych instytucji finansowych.""" + prompt_main
questions_fin = questions_main

PROMPT_FIN = {'prompt': prompt_fin ,'questions': questions_fin}

# ---------- CUKIER ----------
prompt_cukier = """Jesteś ekspertem podatkowym specjalizującym się w zdrowiu publicznym.""" + prompt_main
questions_cukier = questions_main + [
    "Zwróć uwagę na skład napoju, czy dotyczy napoju z dodatkiem cukrów, soków owocowych/warzywnych, substancji słodzących, kofeiny lub tauryny?",
    "Czy wnioskodawca jest podmiotem sprzedającym napoje do punktu sprzedaży detalicznej?",
    "Czy wnioskodawca prowadzi sprzedaż detaliczną napojów?",
    "Czy wnioskodawca jest zamawiającym napój u producenta?"
    ]

PROMPT_CUKIER = {'prompt': prompt_cukier ,'questions': questions_cukier}

# ---------- WIP ----------
prompt_wip = """Jesteś ekspertem podatkowym specjalizującym się w wymianie informacji podatkowych z innymi państwami.""" + prompt_main
questions_wip = questions_main + [
    "Czy wniosek dotyczy automatycznej wymiany o rachunkach raportowanych?",
    "Czy wniosek dotyczy automatycznej wymiany informacji o sprzedawcach?",
    "Czy wniosek dotyczy automatycznej wymiana informacji podatkowych o jednostkach wchodzących w skład grupy podmiotów?"
    ]

PROMPT_WIP = {'prompt': prompt_wip ,'questions': questions_wip}

# ---------- GLOBE ----------
prompt_globe = """Jesteś ekspertem podatkowym specjalizującym się w opodatkowaniu wyrównawczym jednostek składowych grup międzynarodowych i krajowych.""" + prompt_main
questions_globe = questions_main

PROMPT_GLOBE = {'prompt': prompt_globe ,'questions': questions_globe}

# ---------- NIP ----------
prompt_nip = """Jesteś ekspertem podatkowym specjalizującym się w zasadach ewidencji i identyfikacji podatników i płatników (NIP).""" + prompt_main
questions_nip = questions_main

PROMPT_NIP= {'prompt': prompt_nip ,'questions': questions_nip}

# ---------- INNE ----------
prompt_inne = """Jesteś ekspertem podatkowym.""" + prompt_main
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

def tax_prompts(tax_type: str,user_text: str) -> Tuple[str,List[str]]:
    prompt_template = TAX_PROMPTS.get(tax_type, TAX_PROMPTS["inne"])
    questions = prompt_template.get('questions', [])
    prompt = prompt_template.get('prompt', [])
    return prompt.format(user_text=user_text), questions

if __name__ == "__main__":
    pass

    # prompt, questions = tax_prompts(tax_type='vat', user_text='abcd')
    # print(prompt)
    # print(questions)