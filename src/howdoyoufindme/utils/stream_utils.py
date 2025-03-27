# src/howdoyoufindme/utils/stream_utils.py

import json
from typing import Any, Dict

from ..clean_json import clean_and_parse_json


async def create_stream_event(
    event_type: str, message: str = None, task: str = None, data: Dict[str, Any] = None
) -> str:
    """Create a formatted stream event"""
    event = {"type": event_type}
    if message is not None:
        event["message"] = message
    if task is not None:
        event["task"] = task
    if data is not None:
        event["data"] = data
    return json.dumps(event) + "\n"


async def process_task_result(task_name: str, raw_result: str) -> str:
    """Process a task result and create a task_complete event"""
    try:
        # First try to clean and parse the JSON
        try:
            parsed_data = clean_and_parse_json(raw_result)
        except ValueError as e:
            # If cleaning fails, try to extract any valid JSON
            error_msg = str(e)
            # Log the raw output for debugging
            print(f"Raw output that caused error: {raw_result}")
            print(f"JSON parsing error: {error_msg}")
            
            # Try to salvage any valid JSON from the output
            try:
                # Look for JSON-like content
                start_idx = raw_result.find('{')
                end_idx = raw_result.rfind('}')
                if start_idx >= 0 and end_idx > start_idx:
                    json_part = raw_result[start_idx:end_idx + 1]
                    parsed_data = json.loads(json_part)
                else:
                    raise ValueError("No valid JSON structure found")
            except Exception as inner_e:
                # If all recovery attempts fail, return a structured error
                return await create_stream_event(
                    event_type="error",
                    message=f"Failed to process {task_name} output: {error_msg}"
                )

        # If we got this far, we have valid JSON
        return await create_stream_event(
            event_type="task_complete",
            task=task_name,
            data=parsed_data
        )

    except Exception as e:
        return await create_stream_event(
            event_type="error",
            message=f"Unexpected error processing {task_name} output: {str(e)}"
        )