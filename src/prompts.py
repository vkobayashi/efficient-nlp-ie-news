"""
Prompt templates and default schemas:

- ``src/models/prompt_template.py``  (EXTRACT_INSTRUCTION, SUMMARIZE_INSTRUCTION)
- ``src/modules/schema_agent.py``    (the NER / RE / EE / Triple default
  Pydantic-style schemas OneKE prints for its "quick" extraction mode)
- ``src/config.yaml``                (the default per-task instructions)


"""

from __future__ import annotations

# ==================================================================== #
#   Default per-task instructions    #
# ==================================================================== #

DEFAULT_SCHEMA_NOTE = "The final extraction result should be formatted as a JSON object."

TASK_INSTRUCTIONS: dict[str, str] = {
    "NER": "Extract the Named Entities in the given text.",
    "RE": "Extract Relationships between Named Entities in the given text.",
    "EE": "Extract the Events in the given text.",
    "Triple": (
        "Extract the Triples (subject, relation, object) from the given text, "
        "hope that all the relationships for each entity can be extracted."
    ),
}


# -- a stricter, IE-specialist framing used alongside the schema above.
TASK_ROLE_INSTRUCTIONS: dict[str, str] = {
    "NER": (
        "You are an expert in named entity recognition. Please extract entities that match "
        "the schema definition from the input. Return an empty list if the entity type does "
        "not exist. Please respond in the format of a JSON string."
    ),
    "RE": (
        "You are an expert in relationship extraction. Please extract relationship triples "
        "that match the schema definition from the input. Return an empty list for "
        "relationships that do not exist. Please respond in the format of a JSON string."
    ),
    "EE": (
        "You are an expert in event extraction. Please extract events from the input that "
        "conform to the schema definition. Return an empty list for events that do not "
        "exist, and return NAN for arguments that do not exist. If an argument has multiple "
        "values, please return a list. Respond in the format of a JSON string."
    ),
    "Triple": (
        "You are an expert in open information extraction. Please extract subject-relation-"
        "object triples that match the schema definition from the input. Return an empty "
        "list if no triples exist. Please respond in the format of a JSON string."
    ),
}

# schemas for
# quick-mode NER / RE / EE / Triple tasks. Kept verbatim.
TASK_SCHEMAS: dict[str, str] = {
    "NER": """
class Entity(BaseModel):
    name : str = Field(description="The specific name of the entity. ")
    type : str = Field(description="The type or category that the entity belongs to.")
class EntityList(BaseModel):
    entity_list : List[Entity] = Field(description="Named entities appearing in the text.")
""",
    "RE": """
class Relation(BaseModel):
    head : str = Field(description="The starting entity in the relationship.")
    tail : str = Field(description="The ending entity in the relationship.")
    relation : str = Field(description="The predicate that defines the relationship between the two entities.")

class RelationList(BaseModel):
    relation_list : List[Relation] = Field(description="The collection of relationships between various entities.")
""",
    "EE": """
class Event(BaseModel):
    event_type : str = Field(description="The type of the event.")
    event_trigger : str = Field(description="A specific word or phrase that indicates the occurrence of the event.")
    event_argument : dict = Field(description="The arguments or participants involved in the event.")

class EventList(BaseModel):
    event_list : List[Event] = Field(description="The events presented in the text.")
""",
    "Triple": """
class Triple(BaseModel):
    head: str = Field(description="The subject or head of the triple.")
    head_type: str = Field(description="The type of the subject entity.")
    relation: str = Field(description="The predicate or relation between the entities.")
    relation_type: str = Field(description="The type of the relation.")
    tail: str = Field(description="The object or tail of the triple.")
    tail_type: str = Field(description="The type of the object entity.")
class TripleList(BaseModel):
    triple_list: List[Triple] = Field(description="The collection of triples and their types presented in the text.")
""",
}

TASK_LABELS: dict[str, str] = {
    "NER": "Named Entity Recognition (NER)",
    "RE": "Relation Extraction (RE)",
    "EE": "Event Extraction (EE)",
    "Triple": "Open Triple Extraction (subject, relation, object)",
}

# ==================================================================== #
#                    EXTRACT_INSTRUCTION (verbatim)                    #
# ==================================================================== #

EXTRACT_INSTRUCTION = """
**Instruction**: You are an agent skilled in information extarction. {instruction}
{examples}
**Text**: {text}
{additional_info}
**Output Schema**: {schema}

Now please extract the corresponding information from the text. Ensure that the information you extract has a clear reference in the given text. Set any property not explicitly mentioned in the text to null.
"""

SUMMARIZE_INSTRUCTION = """
**Instruction**: Below is a list of results obtained after segmenting and extracting information from a long article. Please consolidate all the answers to generate a final response.

**Task**: {instruction}

**Result List**: {answer_list}
{additional_info}
**Output Schema**: {schema}
Now summarize the information from the Result List.
"""


def build_schema_block(task: str) -> str:
    """default note + printed class schema."""
    return f"{DEFAULT_SCHEMA_NOTE}\n{TASK_SCHEMAS[task]}"


def build_extract_prompt(task: str, text: str) -> str:
    instruction = TASK_ROLE_INSTRUCTIONS[task]
    schema = build_schema_block(task)
    return EXTRACT_INSTRUCTION.format(
        instruction=instruction,
        examples="",
        text=text,
        additional_info="",
        schema=schema,
    )


def build_summarize_prompt(task: str, answer_list: list) -> str:
    instruction = TASK_ROLE_INSTRUCTIONS[task]
    schema = build_schema_block(task)
    return SUMMARIZE_INSTRUCTION.format(
        instruction=instruction,
        answer_list=answer_list,
        additional_info="",
        schema=schema,
    )
