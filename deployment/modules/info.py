INFO = '''
- **cel** &mdash; wyszukiwanie zbliżonych interpretacji z wykorzystaniem metod **sztucznej inteligencji (algorytmy: machine learning, deep learning)**
- **open source** &mdash; wykorzystuje ogólnodostępne biblioteki
- **wersja aplikacji** &mdash; 1.0
- **stan aplikacji** &mdash; wersja w ramach Proof of Concept
- **środowisko** &mdash; maszyna **16 GiB** z systemie operacyjnym **Linux**, baza danych **Postgres**, **Python** w tym biblioteki
- **koszt podstawowy** &mdash; koszt maszyny Linux (serwera), przechowywaniu danych, przesyłu danych i kilku dodatkowych usług w ramach **chmury MF** niezbędnych do działania
- **koszt zmienne** &mdash; **nie wykorzystuje płatnego OPEN AZURE/AI KEY** oraz dziwnych usług Azure - koszty związane z tokenizacją na wejściu, odpytania i na wyjściu **są całkowicie pominięte**
- **gromadzenie danych** &mdash; aplikacja ma ogólnodostępny charakter, nie zbieramy danych osobowych o użytkowniku, **zbierane są jedynie dane historii sesji, modeli AI oraz tabele**]
- **o twórcach** &mdash; przygotowanie interpetacji, modelu ML/DL, analiza danych oraz stworzenie aplikacji zrealizowano dzięki **Ekspertom ds AI w Depatamencie Strategii (Wydział Informacji Zarządzczej)**
'''



FUNKCJE = '''
- **Prompt - wklej pytanie lub opis sprawy (postać)** &mdash; wklejamy treść
- **Wybierz PODATEK** &mdash; zaznaczamy PIT lub VAT, Inne oznacza domyślne ustawienie i zwraca komunikat o zaznaczeniu podatku
- **Wybierz SŁOWA KLUCZOWE** &mdash; z listy wybieramy słowa, zatwierdzając ENTER, wpisując krótką frazę uzyskujemy podpowiedź, max 10 słów, brak słowa oznacza domyślne ustawienie i zwraca komunikat o zaznaczeniu podatku
- **Wyślij (przycisk)** &mdash; wysyłamy żądanie o pobranie danych
- **Asystent (robot)** &mdash; tabela z wyszukanymi podobnymi interpertacjami
- **strzałka przy nazwie kolumny** &mdash; sortowanie po kolumnach
- **chmurka przy nazwie kolumny** &mdash; krótka podpowiedź czym jest kolumna
- **download do CSV** &mdash; pobranie do pliku CSV
- **search** &mdash; wyszukanie w tabeli]
- **fullscreen** &mdash; powiększenie tabeli
- **otwórz link** &mdash; link do interpetacji w https://eureka.mf.gov.pl/
- **ocena (kolorowe twarze)** &mdash; ocena odpowiedzi od negatywnej do pozytywnej
- **opcjonalne** &mdash; tutaj wprowadzamy opis oceny interpertacji]
- **Submit (przycisk)** &mdash; wysyłamy żądanie zapisu oceny interpertacji
- **Wyczyść historię (przycisk)** &mdash; czyścimy historię sesji, tabele oraz **rozpoczynamy nową sesję**
- **Numer sesji** &mdash; unikalny numer sesji
- **Wyloguj** &mdash; opuszczamy aplikację :scream:
'''


PROBLEMY = '''
- **Długo się ładuje** &mdash; problemy z siecią MF

    **Zalecane** &mdash; odświeżyć stronę i zalogować, usunąć pliki cookies
- **Długo się ładuje z powodu długiego tekstu dłuższego interpretacji (więcej niż 1 strona A4)** &mdash; problemy z siecią MF, może spowodować **błąd krytyczny**

    **Zalecane** &mdash; odświeżyć stronę i zalogować, usunąć pliki cookie, **skrócić tekst**
- **Ta strona nie działa Serwer ... nie wysłał żadnych danych. ERR_EMPTY_RESPONSE** &mdash; problemy z siecią MF

    **Zalecane** &mdash; odświeżyć stronę i zalogować, usunąć pliki cookie
- **Your app is having trouble loading the streamlit_feedback.streamlit_feedback component.** &mdash; kłopot z ładowaniem pakietu aplikacji i siecią MF

    **Zalecane** &mdash; kliknij wyczyść historię lub oświeżyć stronę i zalogować, usunąć pliki cookies od aplikacji
- **Brak wpisu treści w prompt. Nie możesz po prostu nic wysłać.** &mdash; brak w **Prompt** znaków

    **Zalecane** &mdash; wpisz w **Prompt** znak lub krótkie zdanie i ponownie **Wyślij**
- **Wybierz typ podatku.** &mdash; brak w nie wybrano typu podatku, **Inne** oznacza wartość domyślną w momencie zmiany sesji lub ponownego uruchomienia

    **Zalecane** &mdash; wybierz w **Wybierz PODATEK** i ponownie **Wyślij*
- **Brak wpisanego słowa kluczowego. Wpisz słowo lub słowa kluczowe.** &mdash; brak w **Wybierz SŁOWA KLUCZOWE** znaków

    **Zalecane** &mdash; wybierz w **Wybierz SŁOWA KLUCZOWE** z listy rozwijanej wpisując frazy (max 10) najbardziej zbliżone dla przykładów kryptowalut: "token”, „waluta wirtualna”, „kryptowaluty” lub dla frazy "zabyt": „ulga na zabytki”, „zabytek”, „przedmiot zabytkowy”, „konserwacja zabytków”, które są podpowiadane, potwierdź wprowadzenie ENTER i ponownie **Wyślij**
- **Brak wyszukania &mdash; brak wektorów/fraz w bazie, niewłaściwy podatek lub wybrane słowa** &mdash; brak fraz do porówniania zapisanych w bazie lub wybór podatku lub słów kluczowych (brak powiązania)

    **Zalecane** &mdash; wybierz właściwy podatek lub słowa kluczowe i ponownie **Wyślij**
- **Komunikaty na czerwono** &mdash; błąd do rozwiązania :woozy_face:

    :red[**Zalecane** &mdash; screen błędu i kontakt]
- **Connection error Connection failed with status 503** &mdash; utrata (kill) sesji serwera, **błąd krytyczny**

    :red[**Zalecane** &mdash; screen błędu i **PILNY** kontakt :woozy_face:]
'''

ZALOZENIA = '''
- :red[**Działanie** &mdash; **wymagane jest deklarowanie** typu podatku oraz słów kluczowych]
- **Ograniczenia** &mdash; Do fazy testów przyjęto liczbę znaków w interpretacji nie przekracza 6400 znaków (ok. 2 strony arkusz A4) oraz minimalna liczba słów 20
- **Działanie** &mdash; działa model AI oparty o wekotryzację tekstu (przypisanie słowom, frazom, tekstom ciągu wartości liczbowym) oraz techniki data mining
- **Działanie** &mdash; model usuwa treści, frazy nie istotne
- **Działanie** &mdash; model rozbijają treści na obszary
- **Działanie** &mdash; model osobno działają na tezach, słownikach, przepisach oraz treściach
- **Działanie** &mdash; model tj. jego pewna część algorytmu przeszukującego działa w oparciu o kombinacje wpisanych słów np.:
    jeśli mamy 3 frazy A, B,C,  model utworzy kombinacje (A,B,C) , (A,B) , (B,C), (A,C) , (A), (B), (C), które będą przeszukiwane od zbioru 3 elementowego do 1 z pewnymi warunkami. 
    Dla 5 fraz będzie to 32 kombinacje, a dla 10 – 1024 – im więcej tym trochę dłuższy czas. 
'''