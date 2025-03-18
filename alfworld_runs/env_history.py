from typing import List, Dict

'''
Environment History object

__init__: attributes
add: add history records
check_is_exhausted: check if the current query is exhausted based on repeated actions
reset: reset the history to an empty state
__str__: convert the environment history to a string representation
_get_base_query: generate the base query string including memory and task details

'''
class EnvironmentHistory:
    """
    A class to manage the history of interactions with an environment.

    Attributes:
    - _cur_query: Current query string that includes base query, starting info, and memory.
    - _history: List of dictionaries recording the interaction history.
    - _last_action: Last action taken in the environment.
    - _is_exhausted: Boolean indicating whether the query is exhausted due to repetitive actions.
    """
    def __init__(self, base_query: str, start_info, memory: List[str], history: List[Dict[str, str]] = []) -> None:
        self._cur_query: str = f'{_get_base_query(base_query, start_info, memory)}'
        self._history: List[Dict[str, str]] = history
        self._last_action: str = ''
        self._is_exhausted: bool = False

    def add(self, label: str, value: str) -> None:
        """
        Add a new record to the interaction history.

        :param label: The type of record to add (must be 'action', 'observation', or 'human_edit').
        :param value: The content of the record.
        """
        assert label in ['action', 'observation', 'human_edit']
        self._history += [{
            'label': label,
            'value': value,
        }]
        if label == 'action':
            if value == self._last_action:
                self._is_exhausted = True
            else:
                self._last_action = value

    def check_is_exhausted(self) -> bool:
        """
        Check whether the query is exhausted (repeated actions detected).

        :return: True if exhausted, False otherwise.
        """
        return self._is_exhausted

    def reset(self) -> None:
        """
        Reset the interaction history to an empty list.
        """
        self._history = []

    def __str__(self) -> str:
        """
        Convert the interaction history into a readable string format.

        :return: A string representation of the environment history.
        """
        s: str = self._cur_query + '\n'
        for i, item in enumerate(self._history):
            if item['label'] == 'action':
                s += f'> {item["value"]}'
            elif item['label'] == 'observation':
                s += item['value']
            # NOT CURRENTLY SUPPORTED
            elif item['label'] == 'human_edit':
                s += f'[human edit]: {item["value"]}'
            if i != len(self._history) - 1:
                s += '\n'
        return s

def _get_base_query(base_query: str, start_info: str, memory: List[str]) -> str:
    """
    Generate the base query string by combining the base query, memory, and task details.

    :param 
    base_query: The base query string.
    start_info: The task description or initial information.
    memory: A list of memory strings to include in the query.

    :return: 
    query: The constructed base query string.
    """
    query = base_query

    # add memory if it exists
    if len(memory) > 0:
        query += '\n\nYour memory for the task below:'
        for i, m in enumerate(memory):
            query += f'\nTrial {i}:\n{m.strip()}'
    query += f"\nHere is the task:\n{start_info}"
    return query
