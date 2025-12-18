import re
from typing import List
# text_norm_fact = 'abc'
# text_norm_q = 'ab'
# print(' '.join([text_norm_fact,text_norm_q]))



# import re
# from typing import List

# def split_by_question_czy_numbered(text: str) -> List[str]:
#     """
#     Split text sequentially by the first occurrences
#     """
#     result = ["", "", "", ""]

#     parts = re.split(r'\d+\.\s*', text)
#     print(parts)
#     q_idx = re.split(r'\?', text)
#     print(q_idx)

#     # 1) Split by first "?"
#     q_idx = text.find("?")
#     print(q_idx)
#     if q_idx == -1:
#         result[0] = text.strip()
#         return result
#     result[0] = text[:q_idx].strip()
#     rest = text[q_idx + 1:]

#     # 2) Split by first "czy" (case-insensitive, as a separate word)
#     m_czy = re.search(r"\bczy\b", rest, flags=re.IGNORECASE)
#     print(m_czy)
#     if not m_czy:
#         result[1] = rest.strip()
#         return result
#     result[1] = rest[:m_czy.start()].strip()
#     rest2 = rest[m_czy.end():]

#     # 3) Split by first numbered-list marker: space + digits + '.' + space
#     m_num = re.search(r"\s\d+\.\s", rest2)
#     print(m_num)
#     #print(m_num.start())
#     if not m_num:
#         result[2] = rest2.strip()
#         return result

#     result[2] = rest2[:m_num.start()].strip()
#     result[3] = rest2[m_num.end():].strip()
#     return result







# def split_by_question_czy_numbered(text: str) -> List[str]:
#     """
#     Split text sequentially by the first occurrences
#     """
#     result = ["", "", "", ""]

#     # 1) Split by first "?"
#     q_idx = text.find("?")
#     if q_idx == -1:
#         result[0] = text.strip()
#         return result
#     result[0] = text[:q_idx].strip()
#     rest = text[q_idx + 1:]

#     # 2) Split by first "czy" (case-insensitive, as a separate word)
#     m_czy = re.search(r"\bczy\b", rest, flags=re.IGNORECASE)
#     if not m_czy:
#         result[1] = rest.strip()
#         return result
#     result[1] = rest[:m_czy.start()].strip()
#     rest2 = rest[m_czy.end():]

#     # 3) Split by first numbered-list marker: space + digits + '.' + space
#     m_num = re.search(r"\s\d+\.\s", rest2)
#     if not m_num:
#         result[2] = rest2.strip()
#         return result

#     result[2] = rest2[:m_num.start()].strip()
#     result[3] = rest2[m_num.end():].strip()
#     result = [q for q  in result if len(q) > 50]
#     return result

# def split_into_batches(data_list, batch_size):
#     """
#     Splits a list into smaller batches of specified size.
    
#     :param data_list: The original list to split
#     :param batch_size: Size of each batch
#     :return: A generator yielding batches
#     """
#     for i in range(0, len(data_list), batch_size):
#         yield data_list[i:i + batch_size]

# # Example usage:
# my_list = [{"id": i, "a":"c"} for i in range(1, 23)]  # Example list of dicts
# batches = list(split_into_batches(my_list, 5))

# for idx, batch in enumerate(batches, start=1):
#     print(batch)

# for batch in split_into_batches(my_list, 5):
#     print(batch)





# def extract_between_stars(text):
#     """
#     Extract all substrings enclosed between ** and **.
    
#     :param text: Input string
#     :return: List of extracted substrings
#     """
#     return re.findall(r"\*\*(.*?)\*\*", text)

# # Example usage:
# sample_text = "This is **bold text** and here is **another one**."
# result = extract_between_stars(sample_text)
# print(result)  # Output: ['bold text', 'another one'



def split_by_question_numbered_czy(text: str):
    text = text[:4420]
    parts = re.split(r'\?|\d+\.\s*|\bczy\s*', text)
    parts = [p.strip() for p in parts if p.strip()]
    return [p for p in parts[1:] if len(p)>3]
    # if len(parts)>0:
    #     return parts
    # parts = re.split(r'\d+\.\s*', text)
    # parts = [p.strip() for p in parts if p.strip()]
    # if len(parts)>2:
    #     return parts[1:]
    # parts = re.split(r'\bczy\s*', text)
    # if len(parts)>0:
    #     return parts


text = "Pytanie   To jest tekst  możemy to zrobić     Tak, możemy.  "
print(split_by_question_numbered_czy(text=text))

# parts = re.split(r'\d+\.\s*', text)
# # Remove empty strings and print results
# parts = [p.strip() for p in parts if p.strip()]
# for i, part in enumerate(parts, start=1):
#     print(f"{i}: {part}")
def split_by_question_numbered_czy(text: str) -> List:
    text = text[:4420]
    parts = re.split(r'\?|\d+\.\s*|\bczy\s*', text)
    parts = [p.strip() for p in parts if p.strip()]
    return [p for p in parts[1:] if len(p)>3]

def extract_between_stars(text) -> List:
    """
    Extract all substrings enclosed between ** and **.
    """
    text = text[:4420]
    return re.findall(r"\*\*(.*?)\*\*", text)
# -----------------------------