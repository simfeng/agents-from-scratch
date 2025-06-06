import pytest
import sys
sys.path.append('..')
from src.eval.email_dataset import email_inputs, expected_tool_calls
from src.utils import format_messages_string
from src.utils import extract_tool_calls
from src.email_assistant import email_assistant

from langsmith import testing as t

@pytest.mark.langsmith
@pytest.mark.parametrize(
    "email_input, expected_tool_calls",
    [
        (email_inputs[0], expected_tool_calls[0]),
        (email_inputs[3], expected_tool_calls[3]),
    ]   
)
def test_email_dataset_tool_calls(email_input, expected_tool_calls):
    """Test if email processing contains expected tool calls.

    这里只测试了工具是否被正确调用，没有测试调用顺序。
    """

    # Run the email assistant
    result = email_assistant.invoke({"email_input": email_input})

    # Extract tool calls from the result
    extracted_tool_calls = extract_tool_calls(result["messages"])

    # check if all expected tool calls are in the extracted ones
    missing_calls = [call for call in expected_tool_calls if call not in extracted_tool_calls]

    t.log_outputs({
        "missing_calls": missing_calls,
        "extracted_tool_calls": extracted_tool_calls,
        "response": format_messages_string(result["messages"])
    })

    assert len(missing_calls) == 0
