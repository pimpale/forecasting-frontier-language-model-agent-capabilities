import contextvars
from typing import Any, TypeVar

from inspect_ai.util import Store

T = TypeVar("T")


class StoreUsageError(Exception):
    pass


def _store_to_string(current_store: Store, indent_tabs: int = 0) -> str:
    output_string = "Store("

    for key, value in current_store.items():
        output_string += f"\n{indent_tabs * '\t'}{key} ({type(value)}): {value}"

    output_string += "\n)"

    return output_string


def _raise_store_usage_error_with_message(current_store: Store, message: str) -> None:
    output_string = _store_to_string(current_store)

    context: contextvars.Context = contextvars.copy_context()

    output_string += "\n\nContext("
    for key, value in context.items():
        output_string += f"\n- {key.name}\n\t - type: {type(value)}\n\t - value:"

        if isinstance(value, Store):
            output_string += f"{_store_to_string(value, indent_tabs=1)}"
        else:
            output_string += f"{value}"

    output_string += "\n)"

    raise StoreUsageError(message + "\n\n" + output_string)


def _get_namespace_for_result(result: Any) -> str:
    return f"{result.__module__}.{result.__class__.__name__}"


def _get_namespace_for_result_type(result_type: type[T]) -> str:
    return f"{result_type.__module__}.{result_type.__name__}"


def store_contains(
    result_type: type[T],
    current_store: Store,
) -> bool:
    namespace = f"{result_type.__module__}.{result_type.__name__}"
    return namespace in current_store


def store_delete(
    result_type: type[T],
    current_store: Store,
) -> None:
    assert current_store is not None, "current_store must be provided"

    namespace = _get_namespace_for_result_type(result_type)

    if namespace not in current_store:
        _raise_store_usage_error_with_message(current_store, f"namespace {namespace} not found in store")

    current_store.delete(namespace)


def store_load(
    result_type: type[T],
    current_store: Store,
) -> T:
    assert current_store is not None, "current_store must be provided"

    namespace = _get_namespace_for_result_type(result_type)

    if namespace not in current_store:
        _raise_store_usage_error_with_message(current_store, f"namespace {namespace} not found in store")

    result_value: T = current_store.get(namespace)

    return result_value


def store_overwrite_existing(
    results: Any,
    current_store: Store,
) -> None:
    assert current_store is not None, "current_store must be provided"

    namespace = _get_namespace_for_result(results)

    if namespace not in current_store:
        _raise_store_usage_error_with_message(current_store, f"namespace {namespace} not found in store")

    current_store.set(namespace, results)


def store_save(
    results: Any,
    current_store: Store,
) -> None:
    assert current_store is not None, "current_store must be provided"

    namespace = _get_namespace_for_result(results)

    if namespace in current_store:
        _raise_store_usage_error_with_message(
            current_store,
            f"namespace {namespace} already exists in store, delete it first via `store_delete` if this is intentional",
        )

    current_store.set(namespace, results)
