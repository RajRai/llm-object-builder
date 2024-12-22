import re
from typing import Any, Dict, List, Union

from ollama import Client


def cond_print(verbose: bool, s: str):
    if verbose:
        print(s)


##############################################################################
# Simple mock LLM
##############################################################################

def mock_llm(query: str) -> str:
    """
    Example LLM-like function. In real usage, you'd replace this with
    Ollama, OpenAI, or any other language model call.
    """
    if "Create a list of questions from the given text:" in query:
        return "What is the capital of France?\nWho wrote 'To Kill a Mockingbird'?"

    if "What is the capital of France?" in query:
        return "Paris\nLyon\nMarseille\nToulouse"

    if "Who wrote 'To Kill a Mockingbird'?" in query:
        return "Harper Lee\nErnest Hemingway\nMark Twain\nF. Scott Fitzgerald"

    return "Unhandled case in mock_llm"


##############################################################################
# Ollama
##############################################################################

def OllamaLLMWrapper(client: Client, model: str, verbose=False):
    def ollama_llm(query: str) -> str:
        cond_print(verbose, f"[LLM] Query:\n{query}\n")
        response = client.generate(model, query).response
        cond_print(verbose, f"[LLM] Response:\n{response}\n")
        return response

    return ollama_llm


##############################################################################
# Find a path in the current node or any of its parents
##############################################################################

def _find_in_parents(path_parts, current_node):
    """
    Given a list of path parts (e.g. ['questionText']),
    we walk the current_node and its _parent chain
    until we find the correct data.

    If we end on a dictionary with a '_value' key,
    we'll return that string instead of the dictionary.
    Otherwise, we return whatever we have (or "" if not found).
    """
    if not path_parts:
        return current_node

    first = path_parts[0]

    # If the first part is in the current node
    if first in current_node:
        val = current_node[first]
        if len(path_parts) == 1:
            # We've reached the final part of the path
            # If 'val' is a dict with '_value', return that string
            if isinstance(val, dict) and "_value" in val:
                return val["_value"]
            else:
                return val
        else:
            # We still have deeper parts to go
            if isinstance(val, dict):
                return _find_in_parents(path_parts[1:], val)
            else:
                # Can't descend into a non-dict
                return ""
    else:
        # If not in current node, try parent
        parent = current_node.get("_parent")
        if isinstance(parent, dict):
            return _find_in_parents(path_parts, parent)
        return ""


##############################################################################
# Resolve placeholders like {parent.questionText} by walking up parents
##############################################################################

def _resolve_query(query: str, node: Dict[str, Any]) -> str:
    """
    Replaces placeholders {some.path} by searching in the current node
    or up through _parent references.
    """
    if not query:
        return ""

    pattern = r"#\{([\w\.]+)\}"

    def replace_match(match):
        path_str = match.group(1)  # e.g. "parent.questionText"
        path_parts = path_str.split(".")
        value = _find_in_parents(path_parts, node)
        return str(value) if value is not None else ""

    return re.sub(pattern, replace_match, query)


##############################################################################
# Build an intermediate structure, storing special keys like _chunk, etc.
##############################################################################

def _build(schema: Dict[str, Any], node: Dict[str, Any], llm) -> None:
    """
    Recursively populate `node` in place according to the schema.
    `node` can have special fields _parent and _chunk for reference.
    """
    # If it's an "object" node: build sub-attributes
    if "attributes" in schema:
        for attr_schema in schema["attributes"]:
            name = attr_schema["name"]

            # Each attribute is a new dictionary to hold sub-data (or possibly a final string)
            # but we also store the parent reference
            child_node = {
                "_chunk": node.get("_chunk", ""),  # pass chunk down (if any)
                "_parent": node,
            }
            node[name] = child_node

            _build(attr_schema, child_node, llm)
        return

    # If it's a "list" node: produce lines and build for each line
    if "listType" in schema:
        query_string = schema.get("queryString", "")
        resolved = _resolve_query(query_string, node)
        resolved += "\nPlace one answer on each line."

        llm_result = llm(resolved)
        lines = llm_result.splitlines()

        # We'll store the list in a special key, then later clean it up
        node["_list"] = []
        for line in lines:
            if line.strip() == '':
                continue
            # Make a sub-node for each line
            child_node = {
                "_parent": node,
                "_chunk": line,
            }
            _build(schema["listType"], child_node, llm)
            node["_list"].append(child_node)
        return

    # Otherwise, treat it as a "string" node
    # 1) if "value", use that
    if "value" in schema:
        node["_value"] = schema["value"]
        return

    # 2) If no queryString, default to chunk
    if "queryString" not in schema:
        node["_value"] = node.get("_chunk", "")
        return

    # 3) Otherwise call LLM
    query_string = schema["queryString"]
    resolved = _resolve_query(query_string, node)
    node["_value"] = llm(resolved)


##############################################################################
# Cleanup the data structure so that it matches the schema
#    (i.e., remove _parent, _chunk, etc.)
##############################################################################

def _cleanup(schema: Dict[str, Any], node: Dict[str, Any]) -> Union[str, Dict[str, Any], List[Any]]:
    """
    Convert the intermediate structure into the final shape:
      - For an object, keep only the named attributes in the schema
      - For a list, return a real list
      - For a string, return the _value
    """
    # If it's an object schema
    if "attributes" in schema:
        cleaned = {}
        for attr_schema in schema["attributes"]:
            name = attr_schema["name"]
            child_data = node.get(name)
            if isinstance(child_data, dict):
                cleaned[name] = _cleanup(attr_schema, child_data)
        return cleaned

    # If it's a list schema
    if "listType" in schema:
        items = node.get("_list", [])
        cleaned_list = []
        for item_dict in items:
            cleaned_list.append(_cleanup(schema["listType"], item_dict))
        return cleaned_list

    # If it's a "string" schema
    # Return whatever was placed in _value (defaults to empty string if missing)
    return node.get("_value", "")


##############################################################################
# Public function to orchestrate everything
##############################################################################

def generate_data(schema: Dict[str, Any], llm, chunk: str = "") -> Union[str, Dict[str, Any], List[Any]]:
    """
    1. Create an empty root node.
    2. Build the entire nested data structure with special fields.
    3. Cleanup the structure, removing all special fields and shaping it.
    """
    root = {
        "_parent": None,
        "_chunk": chunk
    }
    _build(schema, root, llm)
    return _cleanup(schema, root)
