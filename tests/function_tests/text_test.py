# Original dictionary
my_dict = {'old_key': 'value'}

# Rename 'old_key' to 'new_key'
my_dict['new_key'] = my_dict.pop('old_key')

print(my_dict)

def rename_keys(data: dict, key_map: dict) -> dict:
    """
    Rename keys in a dictionary using a mapping.

    Parameters:
    - data: The original dictionary.
    - key_map: A dictionary where keys are old keys and values are new keys.

    Returns:
    - A new dictionary with keys renamed.
    """
    return {key_map.get(k, k): v for k, v in data.items()}
original = {'name': 'Alice', 'age': 30, 'city': 'Warsaw'}
mapping = {'name': 'full_name', 'city': 'location'}

renamed = rename_keys(original, mapping)
print(renamed)

def rename_keys_nested(data: dict, key_map: dict) -> dict:
    """
    Recursively rename keys in a nested dictionary using a mapping.

    Parameters:
    - data: The original dictionary (can be nested).
    - key_map: A dictionary where keys are old keys and values are new keys.

    Returns:
    - A new dictionary with keys renamed.
    """
    if isinstance(data, dict):
        new_dict = {}
        for k, v in data.items():
            new_key = key_map.get(k, k)
            new_dict[new_key] = rename_keys_nested(v, key_map)
        return new_dict
    elif isinstance(data, list):
        return [rename_keys_nested(item, key_map) for item in data]
    else:
        return data

original = {
    'name': 'Alice',
    'details': {
        'age': 30,
        'city': 'Warsaw',
        'contact': {
            'email': 'alice@example.com',
            'phone': '123456789'
        }
    }
}

mapping = {
    'name': 'full_name',
    'city': 'location',
    'email': 'email_address'
}

renamed = rename_keys_nested(original, mapping)
print(renamed)

# {
#     'full_name': 'Alice',
#     'details': {
#         'age': 30,
#         'location': 'Warsaw',
#         'contact': {
#             'email_address': 'alice@example.com',
#             'phone': '123456789'
#         }
#     }
# }