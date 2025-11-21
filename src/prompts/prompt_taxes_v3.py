PROMPT_MAIN = """

Twoim zadaniem jest odpowiedzieć na tekst wnioskodawcy.
Tekst wnioskowadcy: {user_text}.
Oto pytania:
Czy wskazano konkretne przepisy prawa podatkowego? Odpowiedz w formacie JSON z kluczem: przepisy.
Jakie jest 10 słów kluczowych? Odpowiedz w formacie JSON z kluczem: slowa.
Napisz w 5 zdaniach o co pyta wnioskodawca? Odpowiedz w formacie JSON z kluczem: zdania."""


PROMPT_VAT = """
Jesteś ekspertem podatkowym specjalizującym się w podatku od towarów i usług (VAT).

{PROMPT_MAIN}

Czy wnioskodawca jest osobą fizyczną, jednostką samorządu terytorialnego (gmina, powiat, województwo), spółką, czy innym podmiotem (np. stowarzyszenie, spółdzielnia itp.)? Odpowiedz w formacie JSON z kluczem: rodzaj_wnioskodawcy.
Czy wnioskodawca jest czynnym, zarejestrowanym podatnikiem VAT? Odpowiedz w formacie JSON z kluczem: czynny_podatnik_VAT.
Czy wnioskodawca jest zarejestrowany jako podatnik VAT UE? Odpowiedz w formacie JSON z kluczem: zarejestrowany_VAT_UE.
Czy wnioskodawca prowadzi działalność gospodarczą? Odpowiedz w formacie JSON z kluczem: prowadzi_dzialalnosc_gospodarcza.
Czy działalność jest opodatkowana podatkiem VAT, zwolniona, czy mieszana? Odpowiedz w formacie JSON z kluczem: rodzaj_dzialalnosci.
Jaki jest zakres działalności gospodarczej prowadzonej przez wnioskodawcę? Odpowiedz w formacie JSON z kluczem: zakres_dzialalnosci. 
Czy wniosek dotyczy sprzedaży towaru (jakiego)? Odpowiedz w formacie JSON z kluczem: dotyczy_sprzedazy_towaru.
Czy wniosek dotyczy świadczenia usług (jakich)? Odpowiedz w formacie JSON z kluczem: dotyczy_swiadczenia_uslug.
Na podstawie jakiej czynności prawnej wnioskodawca nabył towar? Odpowiedz w formacie JSON z kluczem: podstawa_nabycia_towaru.
Czy wniosek dotyczy eksportu? Czy wniosek dotyczy importu? Odpowiedz w formacie JSON z kluczami: dotyczy_eksportu, dotyczy_importu.
Czy wnioskodawca dokonuje transakcji wewnątrzwspólnotowych (WNT, WDT)? Odpowiedz w formacie JSON z kluczami: dokonuje_WNT, dokonuje_WDT.
Czy przysługiwało prawo do odliczenia podatku VAT od przedmiotowej transakcji?  Odpowiedz w formacie JSON z kluczem: prawo_do_odliczenia_VAT.
Czy nieruchomość jest zabudowana bądź niezabudowana? Odpowiedz w formacie JSON z kluczem: nieruchomosc_zabudowana_niezabudowana.
Czy wartość sprzedaży w roku podatkowym przekroczy kwotę 200.000 zł?  Odpowiedz w formacie JSON z kluczem: wartosc_sprzedazy_pow_200000zl.
Czy dostawa towaru bądź świadczenie usługi korzysta ze zwolnienia? Odpowiedz w formacie JSON z kluczem: korzysta_ze_zwolnienia.
Czy wnioskodawca korzysta w związku z WDT/WNT/eksportem ze stawki podatku VAT w wysokości 0%?  Odpowiedz w formacie JSON z kluczem: stawka_VAT_0_procent.
"""



PROMPT_PIT = """
"Jesteś ekspertem podatkowym specjalizującym się w podatku dochodowym od osób fizycznych (PIT).
Tekst wnioskowadcy: {user_text}.
Twoim zadaniem jest odpowiedzieć na pytania:

Czy wskazano konkretne przepisy prawa podatkowego? Odpowiedz w formacie JSON z kluczem: przepisy.
Jakie jest 10 słów kluczowych? Odpowiedz w formacie JSON z kluczem: slowa_kluczowe.
Napisz w 5 zdaniach o co pyta wnioskodawca? Odpowiedz w formacie JSON z kluczem: zdania.

Czy wnioskodawca jest polskim rezydentem podatkowym? Odpowiedz w formacie JSON z kluczem: rezydencja_podatkowa.
Czy wnioskodawca spełnia warunki rezydencji podatkowej zgodnie z art. 3 ustawy PIT (miejsce zamieszkania, centrum interesów życiowych, długość pobytu)? Odpowiedz w formacie JSON z kluczem: warunki_rezydencji_podatkowej.
Czy wnioskodawca występuje jako podatnik czy jako płatnik? Odpowiedz w formacie JSON z kluczem: rola_wnioskodawcy.
Czy wnioskodawca rozlicza podatek we własnym imieniu, czy działa jako płatnik (np. pracodawca, zleceniodawca)? Odpowiedz w formacie JSON z kluczem: sposob_rozliczenia.
Jakie jest źródło przychodu wnioskodawcy? Odpowiedz w formacie JSON z kluczem: zrodlo_przychodu.
Czy przychód pochodzi z pracy, emerytury, renty, działalności wykonywanej osobiście, pozarolniczej działalności gospodarczej, działalności rolniczej lub działów specjalnych, najmu, kapitałów pieniężnych, zbycia nieruchomości/ruchomości, z zagranicy. Odpowiedz w formacie JSON z kluczem: rodzaj_przychodu.
Czy wnioskodawca jest wspólnikiem spółki? Odpowiedz w formacie JSON z kluczem: jest_wspolnikiem_spolki. 
Określ formę prawną spółki (np. jawna, komandytowa, cywilna) i konsekwencje podatkowe wynikające z uczestnictwa, jeżeli dotyczy. Odpowiedz w formacie JSON z kluczem: forma_prawna_spolki.
Czy została wskazana forma opodatkowania dochodów/przychodów przez wnioskodawcę? Odpowiedz w formacie JSON z kluczem: forma_opodatkowania.
Czy wnioskodawca stosuje skalę podatkową, podatek liniowy, ryczałt od przychodów ewidencjonowanych, kartę podatkową, podatek zryczałtowany, zasady ogólne? Odpowiedz w formacie JSON z kluczem: rodzaj_formy_opodatkowania.
Czy wnioskodawca korzysta z ulg i odliczeń podatkowych? – Zidentyfikuj zastosowane ulgi (np. na dzieci, rehabilitacyjna, termomodernizacyjna, ulga na powrót, na zabytki, mieszkaniowa, dla seniora, na innowacje) oraz ich wpływ na zobowiązanie podatkowe. Odpowiedz w formacie JSON z kluczem: ulgi_i_odliczenia.
"""

PROMPT_CIT = """
"Jesteś ekspertem podatkowym specjalizującym się w podatku dochodowym od osób prawnych (CIT).
Tekst wnioskowadcy: {user_text}.
Twoim zadaniem jest odpowiedzieć na pytania:

Czy wskazano konkretne przepisy prawa podatkowego? Odpowiedz w formacie JSON z kluczem: przepisy.
Jakie jest 10 słów kluczowych? Odpowiedz w formacie JSON z kluczem: slowa_kluczowe.
Napisz w 5 zdaniach o co pyta wnioskodawca? Odpowiedz w formacie JSON z kluczem: streszczenie.

Czy wnioskodawca podlega opodatkowaniu podatkiem dochodowym od osób prawnych?
Czy wnioskodawca jest osobą fizyczną czy osobą prawną?
Jaka jest forma prawna wnioskodawcy?
Czy wnioskodawca podlega w Polsce obowiązkowi podatkowemu od całości swoich dochodów bez względu na miejsce ich osiągania (posiada rezydencję podatkową w Polsce), czy tylko od dochodów, które osiąga na terytorium Polski (nie posiada rezydencji podatkowej w Polsce)?
Czy wnioskodawca występuje w charakterze podatnika czy płatnika?
Czy została wskazana forma opodatkowania dochodów / przychodów? (np. opodatkowanie na zasadach ogólnych, ryczałt od dochodów spółek, podatek u źródła)
Czy wnioskodawca korzysta z ulg podatkowych (np. ulga badawczo-rozwojowa, ulga IP Box, zwolnienie dochodów z tytułu decyzji o wsparciu, itd.)?
Jaki jest zakres działalności prowadzonej przez wnioskodawcę?
Jeżeli któraś z powyższych informacji nie została wskazana wprost, spróbuj ją wywnioskować z treści opisu lub pytań wnioskodawcy.
Odpowiedz w formacie JSON z kluczami:: przepisy, slowa_kluczowe, streszczenie, podlega_opodatkowaniu_CIT, forma_prawna_wnioskodawcy, rezydencja_podatkowa, rola_wnioskodawcy, forma_opodatkowania, ulgi_podatkowe, zakres_dzialalnosci
"""

PROMPT_PCC = """
"Jesteś ekspertem podatkowym specjalizującym się w podatku od czynności cywilnoprawnych.
Tekst wnioskowadcy: {user_text}
Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Czy wniosek dotyczy umowy sprzedaży, umowy zamiany, umowy pożyczki, umowy darowizny, umowy dożywocia, umowy o dział spadku, umowy o zniesienie współwłasności, ustanowienia hipoteki, ustanowienia odpłatnego użytkowania, ustanowienia odpłatnej służebności, umowy depozytu nieprawidłowego, umowy spółki / lub zmiany takich umów?
Czy wniosek dotyczy umowy spółki/zmiany umowy spółki? Jakiego rodzaju spółki?
Czy wniosek dotyczy zasady stand still?
Czy wniosek dotyczy wniesienia wkładu/aportu do spółki? Jakiego rodzaju spółki?
Czy wniosek dotyczy przekształcenia spółki? Jaka spółka będzie przekształcana, a jaką będzie po przekształceniu?
Czy wniosek dotyczy orzeczenia sądu lub ugody?
Czy wniosek dotyczy sprzedaży prawa własności/użytkowania wieczystego nieruchomości?
Czy wniosek dotyczy nabycia prawa własności lokalu mieszkalnego stanowiącego odrębną nieruchomość, prawa własności budynku mieszkalnego jednorodzinnego, spółdzielczego własnościowe prawa do lokalu dotyczącego lokalu mieszkalnego albo domu jednorodzinnego? Jakim tytułem?
Czy wniosek dotyczy gospodarstwa rolnego?
Czy wniosek dotyczy postępowania egzekucyjnego?
Czy wniosek dotyczy cash poolingu?
Czy wniosek dotyczy sprzedaży przedsiębiorstwa lub zorganizowanej części przedsiębiorstwa?
Czy wniosek dotyczy podziału spółki przez wydzielenie lub wyodrębnienie?
Czy wniosek dotyczy walut obcych?
Czy wniosek dotyczy złota dewizowego/złota inwestycyjnego?
Czy wniosek dotyczy spłat lub dopłat?
Jeżeli któraś z powyższych informacji nie została wskazana wprost, spróbuj ją wywnioskować z treści opisu lub pytań wnioskodawcy.
Odpowiedz w formacie JSON z kluczami: przepisy, slowa_kluczowe, streszczenie, dotyczy_umowy_sprzedazy, dotyczy_umowy_zamiany, dotyczy_umowy_pozyczki, dotyczy_umowy_darowizny, dotyczy_umowy_dozywocia, dotyczy_umowy_dzial_spadku, dotyczy_umowy_zniesienie_wspolwlasnosci, dotyczy_ustanowienia_hipoteki, dotyczy_ustanowienia_odplatnego_uzytkowania, dotyczy_ustanowienia_odplatnej_sluzebnosci, dotyczy_umowy_depozytu_nieprawidlowego, dotyczy_umowy_spolki, dotyczy_zasady_stand_still, dotyczy_wniesienia_wkladu_aportu, dotyczy_przeksztalcenia_spolki, dotyczy_orzeczenia_sadu_ugody, dotyczy_sprzedazy_prawa_wlasnosci_uzytkowania_wieczystego, dotyczy_nabycia_prawa_wlasnosci_lokalu_mieszkalnego, dotyczy_gospodarstwa_rolnego, dotyczy_postepowania_egzekucyjnego, dotyczy_cash_poolingu, dotyczy_sprzedazy_przedsiebiorstwa_czesci_przedsiebiorstwa, dotyczy_podzialu_spolki, dotyczy_walut_obcych, dotyczy_zlota_dewizowego_inwestycyjnego, dotyczy_splat_doplat
"""

PROMPT_PSD = """
"Jesteś ekspertem podatkowym specjalizującym się w podatku od spadków i darowizn.
Tekst wnioskowadcy: {user_text}
Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Czy wniosek dotyczy umowy sprzedaży, umowy zamiany, umowy pożyczki, umowy darowizny, umowy dożywocia, umowy o dział spadku, umowy o zniesienie współwłasności, ustanowienia hipoteki, ustanowienia odpłatnego użytkowania, ustanowienia odpłatnej służebności, umowy depozytu nieprawidłowego, umowy spółki / lub zmiany takich umów?
Czy wniosek dotyczy umowy spółki/zmiany umowy spółki? Jakiego rodzaju spółki?
Czy wniosek dotyczy zasady stand still?
Czy wniosek dotyczy wniesienia wkładu/aportu do spółki? Jakiego rodzaju spółki?
Czy wniosek dotyczy przekształcenia spółki?
Jaka spółka będzie przekształcana, a jaką będzie po przekształceniu?
Czy wniosek dotyczy orzeczenia sądu lub ugody?
Czy wniosek dotyczy sprzedaży prawa własności/użytkowania wieczystego nieruchomości?
Czy wniosek dotyczy nabycia prawa własności lokalu mieszkalnego stanowiącego odrębną nieruchomość, prawa własności budynku mieszkalnego jednorodzinnego, spółdzielczego własnościowe prawa do lokalu dotyczącego lokalu mieszkalnego albo domu jednorodzinnego? Jakim tytułem?
Czy wniosek dotyczy gospodarstwa rolnego?
Czy wniosek dotyczy postępowania egzekucyjnego?
Czy wniosek dotyczy cash poolingu?
Czy wniosek dotyczy sprzedaży przedsiębiorstwa lub zorganizowanej części przedsiębiorstwa?
Czy wniosek dotyczy podziału spółki przez wydzielenie lub wyodrębnienie?
Czy wniosek dotyczy walut obcych?
Czy wniosek dotyczy złota dewizowego/złota inwestycyjnego?
Czy wniosek dotyczy spłat lub dopłat?
Jeżeli któraś z powyższych informacji nie została wskazana wprost, spróbuj ją wywnioskować z treści opisu lub pytań wnioskodawcy.
Odpowiedz w formacie JSON z kluczami: przepisy, slowa_kluczowe, streszczenie, dotyczy_umowy_sprzedazy, dotyczy_umowy_zamiany, dotyczy_umowy_pozyczki, dotyczy_umowy_darowizny, dotyczy_umowy_dozywocia, dotyczy_umowy_dzial_spadku, dotyczy_umowy_zniesienie_wspolwlasnosci, dotyczy_ustanowienia_hipoteki, dotyczy_ustanowienia_odplatnego_uzytkowania, dotyczy_ustanowienia_odplatnej_sluzebnosci, dotyczy_umowy_depozytu_nieprawidlowego, dotyczy_umowy_spolki, dotyczy_zasady_stand_still, dotyczy_wniesienia_wkladu_aportu, dotyczy_przeksztalcenia_spolki, dotyczy_orzeczenia_sadu_ugody, dotyczy_sprzedazy_prawa_wlasnosci_uzytkowania_wieczystego, dotyczy_nabycia_prawa_wlasnosci_lokalu_mieszkalnego, dotyczy_gospodarstwa_rolnego, dotyczy_postepowania_egzekucyjnego, dotyczy_cash_poolingu, dotyczy_sprzedazy_przedsiebiorstwa_czesci_przedsiebiorstwa, dotyczy_podzialu_spolki, dotyczy_walut_obcych, dotyczy_zlota_dewizowego_inwestycyjnego, dotyczy_splat_doplat
"""

PROMPT_AKCYZA = """
"Jesteś ekspertem podatkowym specjalizującym się w podatku akcyzowym.
Tekst wnioskowadcy: {user_text}
Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Jakiego rodzaju wyrobu dotyczy wniosek: wyrobu energetycznego, energii elektrycznej, napoju alkoholowego, wyrobu tytoniowego, suszu tytoniowego, płynu do papierosów elektronicznych, wyrobu nowatorskiego, urządzenia do waporyzacji, zestawu części do urządzeń do waporyzacji, saszetek nikotynowych, innych wyrobów nikotynowych albo samochodu osobowego?
Jaka jest klasyfikacja wyrobu akcyzowego albo samochodu osobowego w układzie odpowiadającym Nomenklaturze Scalonej (CN) albo rodzaj wyrobu akcyzowego?
Jakiej czynności dokonuje Wnioskodawca: zakup na terenie Polski, nabycie wewnątrzwspólnotowe, sprzedaż, zużycie, użycie, import, produkcja?
Czy przemieszczanie wyrobu odbywa się w procedurze zawieszenia poboru akcyzy, czy też przemieszczanie wyrobu odbywa się poza procedurą zawieszenia poboru akcyzy?
Czy wyrób jest opodatkowany zerową stawką akcyzy, inną niż zerowa stawka akcyzy czy podlega zwolnieniu od podatku?
Jeżeli któraś z powyższych informacji nie została wskazana wprost, spróbuj ją wywnioskować z treści opisu lub pytań wnioskodawcy.
Odpowiedz w formacie JSON z kluczami: przepisy, slowa_kluczowe, streszczenie, rodzaj_wyrobu, klasyfikacja_wyrobu, czynnosc_dokonuje_wnioskodawca, procedura_przemieszczania_wyrobu, stawka_akcyzy
"""



PROMPT_OP = """
"Jesteś ekspertem podatkowym specjalizującym się w ordynacji podatkowej, e-deklaracjacach oraz ustawy o doręczeniach elektronicznych.
Tekst wnioskowadcy: {user_text}
Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Odpowiedz w formacie JSON z kluczami: przepisy, slowa_kluczowe, streszczenie
"""

PROMPT_GRY = """
"Jesteś ekspertem podatkowym specjalizującym się w grach hazardowych.
Tekst wnioskowadcy: {user_text}
Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Odpowiedz w formacie JSON z kluczami: przepisy, slowa_kluczowe, streszczenie
"""

PROMPT_MALPKI = """
"Jesteś ekspertem podatkowym specjalizującym się w wychowaniu w trzeźwości i przeciwdziałaniu alkoholizmowi.
Tekst wnioskowadcy: {user_text}
Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Czy wnioskodawca sprzedaje napoje alkoholowe w opakowaniach jednostkowych o ilości nominalnej napoju nieprzekraczającej 300 ml?
Jeżeli któraś z powyższych informacji nie została wskazana wprost, spróbuj ją wywnioskować z treści opisu lub pytań wnioskodawcy.
Odpowiedz w formacie JSON z kluczami: przepisy, slowa_kluczowe, streszczenie, sprzedaz_napojow_alcoholowych_opakowaniach_jednostkowych
"""

PROMPT_SPW = """
"Jesteś ekspertem podatkowym specjalizującym się w specjalnym podatku węglowodorowym.
Tekst wnioskowadcy: {user_text}
Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Odpowiedz w formacie JSON z kluczami: przepisy, slowa_kluczowe, streszczenie
"""

PROMPT_PT = """
"Jesteś ekspertem podatkowym specjalizującym się w podateku tonażowym.
Tekst wnioskowadcy: {user_text}
Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Odpowiedz w formacie JSON z kluczami: przepisy, slowa_kluczowe, streszczenie
"""

PROMPT_PKOP = """
"Jesteś ekspertem podatkowym specjalizującym się w podatku od wydobycia niektórych kopalin.
Tekst wnioskowadcy: {user_text}
Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Odpowiedz w formacie JSON z kluczami: przepisy, slowa_kluczowe, streszczenie
"""

PROMPT_SPDET = """
"Jesteś ekspertem podatkowym specjalizującym się w podateku od sprzedaży detalicznej.
Tekst wnioskowadcy: {user_text}
Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Odpowiedz w formacie JSON z kluczami: przepisy, slowa_kluczowe, streszczenie
"""

PROMPT_FIN = """
"Jesteś ekspertem podatkowym specjalizującym się w podatku od niektórych instytucji finansowych.
Tekst wnioskowdacy: {user_text}
Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Odpowiedz w formacie JSON z kluczami: przepisy, slowa_kluczowe, streszczenie
"""

PROMPT_CUKIER = """
"Jesteś ekspertem podatkowym specjalizującym się w zdrowiu publicznym.
Tekst wnioskowadcy: {user_text}
Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Zwróć uwagę na skład napoju, czy dotyczy napoju z dodatkiem cukrów, soków owocowych/warzywnych, substancji słodzących, kofeiny lub tauryny?
Czy wnioskodawca jest podmiotem sprzedającym napoje do punktu sprzedaży detalicznej?
Czy wnioskodawca prowadzi sprzedaż detaliczną napojów?
Czy wnioskodawca jest zamawiającym napój u producenta?
Jeżeli któraś z powyższych informacji nie została wskazana wprost, spróbuj ją wywnioskować z treści opisu lub pytań wnioskodawcy.
Odpowiedz w formacie JSON z kluczami: przepisy, slowa_kluczowe, streszczenie, sklad_napoju, podmiot_sprzedajacy_napoje, prowadzi_sprzedaz_detaliczna, zamawiajacy_napoj
"""

PROMPT_WIP = """
"Jesteś ekspertem podatkowym specjalizującym się w wymianie informacji podatkowych z innymi państwami.
Tekst wnioskowadcy: {user_text}.
Twoim zadaniem jest odpowiedzieć na pytania:

Czy wskazano konkretne przepisy prawa podatkowego? Odpowiedz w formacie JSON z kluczem: przepisy.
Jakie jest 10 słów kluczowych? Odpowiedz w formacie JSON z kluczem: slowa_kluczowe.
Napisz w 5 zdaniach o co pyta wnioskodawca? Odpowiedz w formacie JSON z kluczem: streszczenie.

Czy zapytanie dotyczy automatycznej wymiany o rachunkach raportowanych?
Czy zapytanie dotyczy automatycznej wymiany informacji o sprzedawcach?
Czy zapytanie dotyczy automatycznej wymiana informacji podatkowych o jednostkach wchodzących w skład grupy podmiotów?
Jeżeli któraś z powyższych informacji nie została wskazana wprost, spróbuj ją wywnioskować z treści opisu lub pytań wnioskodawcy.
"""

PROMPT_GLOBE = """
"Jesteś ekspertem podatkowym specjalizującym się w opodatkowaniu wyrównawczym jednostek składowych grup międzynarodowych i krajowych.
Tekst wnioskowadcy: {user_text}.
Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego? Odpowiedz w formacie JSON z kluczem: przepisy.
Jakie jest 10 słów kluczowych? Odpowiedz w formacie JSON z kluczem: slowa.
Napisz w 5 zdaniach o co pyta wnioskodawca? Odpowiedz w formacie JSON z kluczem: zdania.
"""

PROMPT_NIP = """
"Jesteś ekspertem podatkowym specjalizującym się w zasadach ewidencji i identyfikacji podatników i płatników (NIP).
Tekst wnioskowadcy: {user_text}.
Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego? Odpowiedz w formacie JSON z kluczem: przepisy.
Jakie jest 10 słów kluczowych? Odpowiedz w formacie JSON z kluczem: slowa.
Napisz w 5 zdaniach o co pyta wnioskodawca? Odpowiedz w formacie JSON z kluczem: zdania.
"""

PROMPT_INNE = """
"Jesteś ekspertem podatkowym.
Tekst wnioskowadcy: {user_text}.
Twoim zadaniem jest odpowiedzieć na pytania:
O jaki podatek chodzi? Odpowiedz w formacie JSON z kluczem: podatek.
Czy wskazano konkretne przepisy prawa podatkowego? Odpowiedz w formacie JSON z kluczem: przepisy.
Jakie jest 10 słów kluczowych? Odpowiedz w formacie JSON z kluczem: slowa.
Napisz w 5 zdaniach o co pyta wnioskodawca? Odpowiedz w formacie JSON z kluczem: zdania.
"""

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



