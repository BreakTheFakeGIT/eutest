import os
import re
import string
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import nltk
from nltk import bigrams
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import stopwords
import spacy
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from networkx.algorithms import community
import seaborn as sns
import pandas as pd
import numpy as np
from wordcloud import WordCloud, STOPWORDS
    # from PIL import Image

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("pl_core_news_sm")


def visualize_word_pairs(text1, text2, output_folder='./plots/', output_filename='word_pair_comparison.png'):

    tokens1 = word_tokenize(text1.lower())
    tokens2 = word_tokenize(text2.lower())

# # ngramy
# tokeny = tekst.lower().split()
# # tworzenie tabeli tlumaczen
# znaki_do_usuniecia = string.punctuation 
# tabela_tlumaczen = str.maketrans('', '', znaki_do_usuniecia)

# tekst_bez_interpunkcji = tekst_lower.translate(tabela_tlumaczen)
# print(f"Tokeny: {tokeny}")

# n = 2
# ngramy_nltk = list(ngrams(tokeny, n))


    bigrams1 = list(bigrams(tokens1))
    bigrams2 = list(bigrams(tokens2))

    counter1 = Counter(bigrams1)
    counter2 = Counter(bigrams2)

    G = nx.Graph()

    for pair in counter1:
        G.add_edge(pair[0], pair[1], color='blue', weight=1)

    for pair in counter2:
        if G.has_edge(pair[0], pair[1]):
            G[pair[0]][pair[1]]['color'] = 'green'
            G[pair[0]][pair[1]]['weight'] = 2
        else:
            G.add_edge(pair[0], pair[1], color='red', weight=1)

    colors = [G[u][v]['color'] for u, v in G.edges()]
    weights = [G[u][v]['weight'] for u, v in G.edges()]

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, edge_color=colors, width=weights, node_color='lightgray')
    plt.title("Word Pair Comparison: Blue=Text1, Red=Text2, Green=Shared")

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path)
    plt.close()

    print(f"Plot saved to {output_path}")



def draw_bigram_graph(text, output_folder='./plots/', output_filename='netwrokx.png'):
    """
    Draws a bigram frequency graph from the input text.
    """
    # Preprocessing: lowercase and remove punctuation
    text_lower = text.lower()
    translation_table = str.maketrans('', '', string.punctuation)
    text_no_punctuation = text_lower.translate(translation_table)
    tokens = text_no_punctuation.split()

    print(f"Tokens: {tokens}")

    # Generate bigrams
    n = 2
    bigrams_list = list(ngrams(tokens, n))
    print(f"\nGenerated {n}-grams (bigrams):")
    print(bigrams_list)

    # Count bigram occurrences
    bigram_counter = Counter(bigrams_list)

    # Filter bigrams with frequency > 0
    bigrams_above_1 = {
        pair: count
        for pair, count in bigram_counter.items()
        if count > 0
    }

    print("Frequency of each bigram:")
    for pair, count in bigrams_above_1.items():
        print(f"{pair}: {count}")

    # Create edge list with weights
    edge_list = [
        (word1, word2, {"weight": weight})
        for (word1, word2), weight in bigrams_above_1.items()
    ]

    # Create graph
    G = nx.Graph()
    G.add_edges_from(edge_list)

    # Get edge weights and set line thickness
    edge_labels = nx.get_edge_attributes(G, "weight")
    weights = [data['weight'] for _, _, data in G.edges(data=True)]
    line_thickness = [weight * 0.4 for weight in weights]

    # Draw graph
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(30, 28))
    plt.axis('off')

    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=line_thickness, edge_color='darkslategray', alpha=0.7)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', font_size=9)

    plt.title("Bigram Frequency Graph", fontsize=16)
    plt.show()
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path)
    plt.close()




def group_text_by_category_with_plot(texts, similarity_threshold=0.3, output_folder='./plots/', output_filename="text_graph.png"):
    """
    Groups text items by category using NetworkX and cosine similarity.
    Also saves a plot of the graph with communities highlighted.

    Parameters:
    - texts: List of strings.
    - similarity_threshold: Minimum similarity to create an edge.
    - plot_filename: Filename to save the graph plot.

    Returns:
    - List of sets, each representing a group of similar texts.
    """
    # Vectorize the text
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Compute similarity matrix
    similarity_matrix = cosine_similarity(tfidf_matrix)

    # Build graph
    G = nx.Graph()
    for i, text in enumerate(texts):
        G.add_node(i, label=text)

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if similarity_matrix[i][j] >= similarity_threshold:
                G.add_edge(i, j, weight=similarity_matrix[i][j])

    # Detect communities
    communities = community.greedy_modularity_communities(G)

    # Assign colors to communities
    color_map = {}
    for idx, comm in enumerate(communities):
        for node in comm:
            color_map[node] = idx

    node_colors = [color_map.get(node, 0) for node in G.nodes]

    # Plot the graph
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'),
            node_color=node_colors, cmap=plt.cm.Set3, node_size=500, font_size=8)
    plt.title("Text Similarity Graph with Communities")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path)
    plt.close()

    # Return grouped texts
    grouped_texts = [[texts[i] for i in group] for group in communities]
    return grouped_texts


import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import community  # install with: pip install python-louvain

# Sample text
text = """
Interpretacja indywidualna
– stanowisko nieprawidłowe
Szanowni Państwo,
stwierdzam, że Państwa stanowisko w sprawie oceny skutków podatkowych opisanego zdarzenia przyszłego w podatku od towarów i usług jest nieprawidłowe.
Zakres wniosku o wydanie interpretacji indywidualnej
28 lipca 2025 r. wpłynął Państwa wniosek z 28 lipca 2025 r. o wydanie interpretacji indywidualnej dotyczący opodatkowania paczek wydanych w ramach kampanii reklamowej. Uzupełnili go Państwo – w odpowiedzi na wezwanie – pismem z 4 września 2025 r. (wpływ 4 września 2025 r.).
Treść wniosku jest następująca:
Opis zdarzenia przyszłego
Wnioskodawca w swojej działalności prowadzi sprzedaż towarów, kosmetyków i wyrobów medycznych oraz przeprowadza szkolenia z zakresu zabiegów kosmetycznych.
Wnioskodawca zamierza przeprowadzić kampanię reklamową polegającą na przekazywaniu paczek z produktami o wartości do 700 zł brutto. Paczki z produktami mają być opatrzone logo firmy i przekazane do losowo wybranych osób fizycznych, które nie są jego klientami ani kontrahentami. Na zawartość paczki mają składać się pełnowartościowe towary handlowe i dodatkowo gadżety takie jak np. klamra do włosów, pomadka itd. Celem jest promocja firmy, zwiększenie świadomości marki oraz pozyskiwanie przyszłych klientów.
Uzupełnienie i doprecyzowanie opisu zdarzenia przyszłego
Na pytanie 1: Czy są i będą Państwo na dzień przekazania paczek z produktami zarejestrowani jako czynny podatnik podatku od towarów i usług, odpowiedzieli Państwo: Tak.
Na pytanie 2: Czy paczki z produktami będą przez Państwa przekazywane nieodpłatnie, odpowiedzieli Państwo: Tak.
Na pytanie 3: Czy w zamian za przekazanie paczek otrzymają Państwo jakiekolwiek wynagrodzenie lub osoby obdarowane zobowiązane będą do wykonania określonych czynności? Jeśli tak, proszę wskazać jakich czynności, odpowiedzieli Państwo: Nie.
Na pytanie 4: Czy w odniesieniu do przekazywanych towarów przysługiwało Państwu/będzie Państwu przysługiwało prawo do obniżenia kwoty podatku należnego o kwotę podatku naliczonego z tytułu ich nabycia, importu lub wytworzenia tych towarów lub ich części składowych, odpowiedzieli Państwo: Tak.
Na pytanie 5: Czy przekazanie paczek z produktami w ramach kampanii reklamowej nastąpi wyłącznie na cele związane z Państwa działalnością gospodarczą, odpowiedzieli Państwo: Tak.
Na pytanie 6: Czy przekazywane produkty będą związane z prowadzoną przez Państwa działalnością gospodarczą? W jaki sposób, odpowiedzieli Państwo: Tak, są to towary handlowe nabywane w celu dalszej odsprzedaży. Dodatkowo paczki będą zawierały gadżety takie jak klamra do włosów, pomadka. Z naszych towarów handlowych i dodatkowych gadżetów będzie przygotowywana paczka PR-owa.
Na pytanie 7: Jaką funkcję pełnią przekazywane produkty, np. funkcję promocyjną, informacyjną, itp., odpowiedzieli Państwo: Przekazywane produkty mają pełnić funkcję promocyjną, zwiększyć świadomość marki i pozyskać przyszłych klientów.
Na pytanie 8: Czy przekazywane produkty mają wartość konsumpcyjną lub też użytkową dla konsumentów, odpowiedzieli Państwo: Tak.
Na pytanie 9: Czy będą Państwo prowadzili ewidencję pozwalającą na ustalenie tożsamości odbiorców, odpowiedzieli Państwo: Nie.
Na pytanie 10: Czy na paczki z produktami będą się składały prezenty o małej wartości, tj. przekazywane jednej osobie towary:
1) łącznej wartości nieprzekraczającej w roku podatkowym kwoty 100 zł (bez podatku), jeżeli prowadzą Państwo ewidencję pozwalającą na ustalenie tożsamości obdarowanych osób;
2) których przekazania nie ujęto w ewidencji, jeżeli jednostkowa cena nabycia towaru (bez podatku), a gdy nie ma ceny nabycia, jednostkowy koszt wytworzenia, określone w momencie przekazywania towaru, nie przekraczają 20 zł,
odpowiedzieli Państwo: Nie.
Na pytanie 11: Czy produkty przekazywane w paczkach będą stanowiły próbki, przez które należy rozumieć egzemplarz towaru lub jego niewielką ilość, które pozwalają na ocenę cech i właściwości towaru w jego końcowej postaci? Czy ewentualne przekazanie próbki będzie miało na celu promocję tego towaru, odpowiedzieli Państwo: Nie.
Na pytanie 12: Czy przekazywana próbka nie będzie służyła zasadniczo zaspokojeniu potrzeb odbiorcy końcowego w zakresie przekazywanego towaru? A jeśli będzie, to czy zaspokojenie potrzeb tego odbiorcy jest nieodłącznym elementem promocji tego towaru i ma skłaniać tego odbiorcę do zakupu promowanego towaru, odpowiedzieli Państwo: Nie będą to próbki.
"""

# Step 1: Build a co-occurrence graph of words
def build_cooccurrence_graph(text, window_size=2):
    words = text.lower().split()
    G = nx.Graph()
    for i in range(len(words)):
        for j in range(i+1, min(i+window_size+1, len(words))):
            if words[i] != words[j]:
                G.add_edge(words[i], words[j])
    return G

# Step 2: Analyze and visualize
def analyze_text_graph(G,output_folder='./plots/', output_filename="ntx_text_graph.png"):
    # Centrality metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    pagerank = nx.pagerank(G)

    # Louvain clustering
    partition = community.best_partition(G)

    # Visualization
    pos = nx.spring_layout(G)
    cmap = plt.get_cmap("viridis")
    plt.figure(figsize=(10, 7))
    nx.draw_networkx_nodes(G, pos, node_size=500,
                           cmap=cmap, node_color=list(partition.values()))
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title("Text Graph with Louvain Clusters")
    plt.axis("off")
    #plt.show()
    plt.title("Text Similarity Graph with Communities")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path)
    plt.close()



    return {
        "degree_centrality": degree_centrality,
        "betweenness_centrality": betweenness_centrality,
        "closeness_centrality": closeness_centrality,
        "pagerank": pagerank,
        "clusters": partition
    }




# Constants
FILTERED_POS = {"NOUN", "VERB", "ADJ"}
CO_OCCURRENCE_WINDOW = 3  # Window in which words are considered related

def build_cooccurrence_graph(text, pos_tags=FILTERED_POS, window_size=CO_OCCURRENCE_WINDOW):
    """
    Processes the input text, filters by POS tags, and builds a co-occurrence graph.
    
    Parameters:
        text (str): Input text in Polish.
        pos_tags (set): POS tags to include (e.g., {"NOUN", "VERB", "ADJ"}).
        window_size (int): Size of the co-occurrence window.
    
    Returns:
        networkx.Graph: A weighted co-occurrence graph.
    """
    try:
        nlp = spacy.load("pl_core_news_sm")
    except OSError:
        print("Error: Model 'pl_core_news_sm' not found. Download it using:")
        print("python -m spacy download pl_core_news_sm")
        return None

    doc = nlp(text)
    filtered_lemmas = [
        token.lemma_ for token in doc
        if token.pos_ in pos_tags and not token.is_punct and len(token.text) >= 2
    ]

    # Create edges based on co-occurrence
    edges = []
    for i in range(len(filtered_lemmas)):
        for j in range(i + 1, min(i + 1 + window_size, len(filtered_lemmas))):
            u, v = sorted([filtered_lemmas[i], filtered_lemmas[j]])
            if u != v:
                edges.append((u, v))

    edge_counts = Counter(edges)
    edge_list = [(u, v, {"weight": count}) for (u, v), count in edge_counts.items() if count > 1]

    # Build graph
    G = nx.Graph()
    G.add_edges_from(edge_list)
    return G

def visualize_graph(G, title="Co-occurrence Graph", output_folder='./plots/', output_filename="ntx_text_graph.png"):
    """
    Visualizes the given graph with edge weights.
    
    Parameters:
        G (networkx.Graph): The graph to visualize.
        title (str): Title of the plot.
    """
    if G is None or G.number_of_edges() == 0:
        print("Graph is empty or None.")
        return

    pos = nx.spring_layout(G, seed=42)
    weights = [d['weight'] for (_, _, d) in G.edges(data=True)]
    edge_widths = [w * 0.4 for w in weights]
    labels = nx.get_edge_attributes(G, "weight")

    plt.figure(figsize=(30, 28))
    plt.axis('off')
    nx.draw_networkx_nodes(G, pos, node_size=1500, node_color='skyblue', alpha=0.8)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='darkslategray', alpha=0.7)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_color='red', font_size=9)
    plt.title(title, fontsize=20)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, output_filename)
    plt.savefig(output_path)
    plt.close()




# Mapping of POS tags to Polish names
POLISH_POS = {
    "ADJ": "Przymiotnik",
    "ADP": "Przyimek",
    "ADV": "Przysłówek",
    "AUX": "Czasownik posiłkowy",
    "CONJ": "Spójnik",
    "CCONJ": "Spójnik współrzędny",
    "DET": "Zaimek",
    "INTJ": "Wykrzyknik",
    "NOUN": "Rzeczownik",
    "NUM": "Liczebnik",
    "PART": "Partykuła",
    "PRON": "Zaimek",
    "PROPN": "Nazwa własna",
    "PUNCT": "Interpunkcja",
    "SCONJ": "Spójnik podrzędny",
    "SYM": "Symbol",
    "VERB": "Czasownik",
    "X": "Inne",
    "SPACE": "Spacja"
}

def process_text_and_plot(text, word_limit=50,output_folder='./plots/', output_filename="nrr545456tx_text_graph.png"):
    doc = nlp(text)
    pos_data = []

    # Filter and collect POS + lemma
    for token in doc:
        if token.is_stop or token.is_punct or token.is_space or len(token.text) < 3:
            continue
        pos_tag = token.pos_
        pos_name = POLISH_POS.get(pos_tag, pos_tag)
        pos_data.append((pos_name, token.lemma_))

    # Count frequency of (POS, lemma) pairs
    pos_counts = Counter(pos_data)

    # Prepare DataFrame for plotting
    df = pd.DataFrame([
        {'Część Mowy': pos, 'Lemat': lemma, 'Waga': count}
        for (pos, lemma), count in pos_counts.items()
    ])

    unique_pos = df['Część Mowy'].unique()
    unique_pos = [pos for pos in unique_pos if pos not in ['X', 'Liczebnik']]

    # Plot top lemmas per POS category
    for pos_category in unique_pos:
        df_filtered = df[df['Część Mowy'] == pos_category].sort_values(by='Waga', ascending=False).head(word_limit)
        if df_filtered.empty:
            continue

        width = 7
        height = len(df_filtered) * 0.3 + 1

        plt.figure(figsize=(width, height))
        sns.barplot(
            x='Waga',
            y='Lemat',
            data=df_filtered,
            color=sns.color_palette("viridis")[0]
        )

        plt.title(f'TOP {word_limit} Most Frequent Words: {pos_category}')
        plt.xlabel('Frequency')
        plt.ylabel('Lemma')
        plt.xlim(0, max(df_filtered['Waga']) * 1.1)
        plt.tight_layout()
        plt.show()
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, output_filename)
        plt.savefig(output_path)
        plt.close()




process_text_and_plot(text=text)





# graph = build_cooccurrence_graph(text)
# visualize_graph(graph)


# # Run the analysis
# G = build_cooccurrence_graph(text)
# results = analyze_text_graph(G)
# print(results)

# text1 = "This is a simple example to show how words connect."
# text2 = "This example shows how different words can connect."

# #visualize_word_pairs(text1, text2)
# draw_bigram_graph(text=text1)



# texts = [
#     "Apple releases new iPhone",
#     "Samsung unveils Galaxy phone",
#     "Apple stock rises",
#     "Samsung stock falls",
#     "New features in iOS",
#     "Android update released"
# ]

# groups = group_text_by_category_with_plot(texts)
