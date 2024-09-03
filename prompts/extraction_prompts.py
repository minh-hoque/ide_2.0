EXTRACT_PROMPT = """You are an expert in natural language processing and entity extraction. Your task is to extract specific entities from the given text and provide the results in a structured JSON format.

Text: {text}

Entities to extract: policy_number, named_insured, effective_date
--------------
Instructions:
1. Carefully read the provided text.
2. Identify and extract the requested entities from the text.
3. For each entity, provide the exact text as it appears in the original text.
4. If an entity is not present in the text, include an empty list for that entity.
5. If there are multiple instances of an entity, include all occurrences in the list.
6. Present your findings in a JSON format where the keys are the entity names and the values are lists of extracted values.
--------------
Please provide your extraction results in the following JSON format:

```json
{{
  "entity1": ["extracted value 1", "extracted value 2", ...],
  "entity2": ["extracted value 1", "extracted value 2", ...],
  ...
}}
```
--------------
If no instances of an entity are found, use an empty list: "entity_name": []
Remember to be precise and thorough in your extraction. If you're unsure about an entity, provide your best interpretation based on the context.
--------------
extraction:
"""

EXTRACT_PROMPT_2 = """You are an expert in natural language processing and entity extraction. Your task is to extract specific entities from the given text and provide the results in a structured JSON format.

Text: {text}

Entities to extract: policy_number, named_insured, effective_date

Entity Definitions:
- policy_number: The unique identifier assigned to an insurance policy
- named_insured: The individual or entity listed as the primary insured on the policy
- effective_date: The date on which the insurance policy becomes active or goes into effect

--------------
Instructions:
1. Carefully read the provided text.
2. Identify and extract the requested entities from the text.
3. For each entity, provide the exact text as it appears in the original text.
4. If an entity is not present in the text, include an empty list for that entity.
5. If there are multiple instances of an entity, include all occurrences in the list.
6. Present your findings in a JSON format where the keys are the entity names and the values are lists of extracted values.
--------------
Please provide your extraction results in the following JSON format:

```json
{{
  "entity1": ["extracted value 1", "extracted value 2", ...],
  "entity2": ["extracted value 1", "extracted value 2", ...],
  ...
}}
```
--------------
If no instances of an entity are found, use an empty list: "entity_name": []
Remember to be precise and thorough in your extraction. If you're unsure about an entity, provide your best interpretation based on the context.
--------------
extraction:
"""
# Entities to extract: {{entities}}
