PROMPT_VAT = """
"Jesteś ekspertem podatkowym specjalizującym się:
w podatku od towarów i usług (VAT),
w odliczaniu lub zwrotu kwot wydanych na zakup kas rejestrujących oraz zwrotu tych kwot przez podatnika,
w miejsc świadczenia usług oraz zwrotu kwoty podatku naliczonego jednostce dokonującej nabycia (importu) towarów lub usług,
w zwolnień od podatku od towarów i usług oraz warunków stosowania tych zwolnień,
w towarów i usług, dla których obniża się stawkę podatku od towarów i usług, oraz warunków stosowania stawek obniżonych,
w zwrotu podatku od towarów i usług siłom zbrojnym, wielonarodowym kwaterom i dowództwom, towarzyszącemu im personelowi cywilnemu, członkom ich rodzin oraz jednostkom dokonującym nabycia towarów lub usług na rzecz sił zbrojnych
w przypadków i trybu zwrotu podatku od towarów i usług siłom zbrojnym, wielonarodowym kwaterom i dowództwom oraz ich personelowi cywilnemu
w sprawie sposobu określania zakresu wykorzystywania nabywanych towarów i usług do celów działalności gospodarczej w przypadku niektórych podatników 
w pojazdów samochodowych uznawanych za wykorzystywane wyłącznie do działalności gospodarczej podatnika,
w sprawie przypadków, w których nie stosuje się warunku prowadzenia ewidencji przebiegu pojazdu,
w wspólnego systemu podatku od wartości dodanej,
w sprawie wystawiania faktur,
w jednolitego pliku kontrolnego,
w sprawie szczegółowego zakresu danych zawartych w deklaracjach podatkowych i w ewidencji w zakresie podatku od towarów i usług,
w wspólnego systemu podatku od wartości dodanej.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Czy wnioskodawca jest osobą fizyczną, jednostką samorządu terytorialnego (gmina, powiat, województwo), spółką, czy innym podmiotem (np. stowarzyszenie, spółdzielnia itp.)?
Czy wnioskodawca jest czynnym, zarejestrowanym podatnikiem VAT?
Czy wnioskodawca jest zarejestrowany jako podatnik VAT UE?
Czy wnioskodawca prowadzi działalność gospodarczą?
Czy działalność jest opodatkowana podatkiem VAT, zwolniona, czy mieszana?
Jaki jest zakres działalności gospodarczej prowadzonej przez wnioskodawcę?
Czy wniosek dotyczy sprzedaży towaru (jakiego)?
Czy wniosek dotyczy świadczenia usług (jakich)?
Na podstawie jakiej czynności prawnej wnioskodawca nabył towar?
Czy wniosek dotyczy eksportu? Czy wniosek dotyczy importu?
Czy wnioskodawca dokonuje transakcji wewnątrzwspólnotowych (WNT, WDT)?
Czy przysługiwało prawo do odliczenia podatku VAT od przedmiotowej transakcji?
Czy nieruchomość jest zabudowana bądź niezabudowana?
Czy wartość sprzedaży w roku podatkowym przekroczy kwotę 200.000 zł?
Czy dostawa towaru bądź świadczenie usługi korzysta ze zwolnienia?
Czy wnioskodawca korzysta w związku z WDT/WNT/eksportem ze stawki podatku VAT w wysokości 0%?
Jeżeli któraś z powyższych informacji nie została wskazana wprost, spróbuj ją wywnioskować z treści opisu lub pytań wnioskodawcy.

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_PIT = """
"Jesteś ekspertem podatkowym specjalizującym się:
w podatku dochodowym od osób fizycznych (PIT),
w prowadzeniu ewidencji przychodów i wykazu środków trwałych oraz wartości niematerialnych i prawnych,
w zwrocie osobom fizycznym niektórych wydatków związanych z budownictwem mieszkaniowym,
oraz pomocy państwa w nabyciu pierwszego mieszkania przez młodych ludzi.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Czy wnioskodawca jest polskim rezydentem podatkowym?
Czy wnioskodawca spełnia warunki rezydencji podatkowej zgodnie z art. 3 ustawy PIT (miejsce zamieszkania, centrum interesów życiowych, długość pobytu),
Czy wnioskodawca występuje jako podatnik czy jako płatnik?
Czy wnioskodawca rozlicza podatek we własnym imieniu, czy działa jako płatnik (np. pracodawca, zleceniodawca)?
Jakie jest źródło przychodu wnioskodawcy?
Czy przychód pochodzi z pracy, emerytury, renty, działalności wykonywanej osobiście, pozarolniczej działalności gospodarczej, działalności rolniczej lub działów specjalnych, najmu, kapitałów pieniężnych, zbycia nieruchomości/ruchomości, z zagranicy,
Czy wnioskodawca jest wspólnikiem spółki?
Określ formę prawną spółki (np. jawna, komandytowa, cywilna) i konsekwencje podatkowe wynikające z uczestnictwa,
Czy została wskazana forma opodatkowania dochodów/przychodów przez wnioskodawcę?
Czy wnioskodawca stosuje skalę podatkową, podatek liniowy, ryczałt od przychodów ewidencjonowanych, kartę podatkową, podatek zryczałtowany,
Czy wnioskodawca korzysta z ulg i odliczeń podatkowych? – Zidentyfikuj zastosowane ulgi (np. na dzieci, rehabilitacyjna, termomodernizacyjna, ulga na powrót, na zabytki, mieszkaniowa, dla seniora, na innowacje) oraz ich wpływ na zobowiązanie podatkowe.
Jeżeli któraś z powyższych informacji nie została wskazana wprost, spróbuj ją wywnioskować z treści opisu lub pytań wnioskodawcy.

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_CIT = """
"Jesteś ekspertem podatkowym specjalizującym się:
w podatku dochodowym od osób prawnych (CIT),
w cenach transferowych w zakresie podatku dochodowego od osób prawnych,
w sposobie i trybu eliminowania podwójnego opodatkowania w przypadku korekty zysków podmiotów powiązanych w zakresie podatku dochodowego od osób prawnych,
w informacji zawartych w dokumentacji podatkowej w zakresie podatku dochodowego od osób prawnych,
w gospodarce finansowej przedsiębiorstw państwowych,
oraz w wpłatach z zysków przez jednoosobowe spółki Skarbu Państwa.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Czy wnioskodawca podlega opodatkowaniu podatkiem dochodowym od osób prawnych?
Czy wnioskodawca jest osobą fizyczną czy osobą prawną?
Jaka jest forma prawna wnioskodawcy?
Czy wnioskodawca podlega w Polsce obowiązkowi podatkowemu od całości swoich dochodów bez względu na miejsce ich osiągania (posiada rezydencję podatkową w Polsce), czy tylko od dochodów, które osiąga na terytorium Polski (nie posiada rezydencji podatkowej w Polsce)?
Czy wnioskodawca występuje w charakterze podatnika czy płatnika?
Czy została wskazana forma opodatkowania dochodów / przychodów? (np. opodatkowanie na zasadach ogólnych, ryczałt od dochodów spółek, podatek u źródła)
Czy wnioskodawca korzysta z ulg podatkowych (np. ulga badawczo-rozwojowa, ulga IP Box, zwolnienie dochodów z tytułu decyzji o wsparciu, itd.)?
Jaki jest zakres działalności prowadzonej przez wnioskodawcę?
Jeżeli któraś z powyższych informacji nie została wskazana wprost, spróbuj ją wywnioskować z treści opisu lub pytań wnioskodawcy.

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_PCC = """
"Jesteś ekspertem podatkowym specjalizującym się w podatku od czynności cywilnoprawnych.

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

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_PSD = """
"Jesteś ekspertem podatkowym specjalizującym się w podatku od spadków i darowizn.

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

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_AKCYZA = """
"Jesteś ekspertem podatkowym specjalizującym się w podatku akcyzowym.

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

Tekst wnioskowadcy: {user_text}"
"""



PROMPT_OP = """
"Jesteś ekspertem podatkowym specjalizującym się"
w ordynacji podatkowej
e-deklaracjach
oraz ustawy o doręczeniach elektronicznych.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_GRY = """
"Jesteś ekspertem podatkowym specjalizującym się w grach hazardowych.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_MALPKI = """
"Jesteś ekspertem podatkowym specjalizującym się w wychowaniu w trzeźwości i przeciwdziałaniu alkoholizmowi.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Czy wnioskodawca sprzedaje napoje alkoholowe w opakowaniach jednostkowych o ilości nominalnej napoju nieprzekraczającej 300 ml?
Jeżeli któraś z powyższych informacji nie została wskazana wprost, spróbuj ją wywnioskować z treści opisu lub pytań wnioskodawcy.

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_SPW = """
"Jesteś ekspertem podatkowym specjalizującym się w specjalnym podatku węglowodorowym.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_PT = """
"Jesteś ekspertem podatkowym specjalizującym się w podateku tonażowym.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_PKOP = """
"Jesteś ekspertem podatkowym specjalizującym się w podatku od wydobycia niektórych kopalin.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_SPDET = """
"Jesteś ekspertem podatkowym specjalizującym się w podateku od sprzedaży detalicznej.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_FIN = """
"Jesteś ekspertem podatkowym specjalizującym się w podatku od niektórych instytucji finansowych.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_CUKIER = """
"Jesteś ekspertem podatkowym specjalizującym się w zdrowiu publicznym.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Zwróć uwagę na skład napoju, czy dotyczy napoju z dodatkiem cukrów, soków owocowych/warzywnych, substancji słodzących, kofeiny lub tauryny?
Czy wnioskodawca jest podmiotem sprzedającym napoje do punktu sprzedaży detalicznej?
Czy wnioskodawca prowadzi sprzedaż detaliczną napojów?
Czy wnioskodawca jest zamawiającym napój u producenta?
Jeżeli któraś z powyższych informacji nie została wskazana wprost, spróbuj ją wywnioskować z treści opisu lub pytań wnioskodawcy.

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_WIP = """
"Jesteś ekspertem podatkowym specjalizującym się w wymianie informacji podatkowych z innymi państwami.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?
Czy zapytanie dotyczy automatycznej wymiany o rachunkach raportowanych?
Czy zapytanie dotyczy automatycznej wymiany informacji o sprzedawcach?
Czy zapytanie dotyczy automatycznej wymiana informacji podatkowych o jednostkach wchodzących w skład grupy podmiotów?
Jeżeli któraś z powyższych informacji nie została wskazana wprost, spróbuj ją wywnioskować z treści opisu lub pytań wnioskodawcy.

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_GLOBE = """
"Jesteś ekspertem podatkowym specjalizującym się w opodatkowaniu wyrównawczym jednostek składowych grup międzynarodowych i krajowych.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_NIP = """
"Jesteś ekspertem podatkowym specjalizującym się w zasadach ewidencji i identyfikacji podatników i płatników (NIP).

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?

Tekst wnioskowadcy: {user_text}"
"""

PROMPT_INNE = """
"Jesteś ekspertem podatkowym.

Twoim zadaniem jest odpowiedzieć na pytania:
Czy wskazano typ podatku?
Czy wskazano konkretne przepisy prawa podatkowego?
Jakie jest 10 słów kluczowych?
Napisz w 5 zdaniach o co pyta wnioskodawca?

Tekst wnioskowadcy: {user_text}"
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
