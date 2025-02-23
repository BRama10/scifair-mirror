import re

def extract_tag_content(text: str, tag_name: str) -> list[str]:
    """
    Extract content within specified XML-like tags.
    
    Args:
        text (str): The input text containing XML-like tags
        tag_name (str): The name of the tag to extract content from
        
    Returns:
        list[str]: List of all matches found within the specified tags
        
    Example:
        >>> text = "hello <answer>yes</answer> <answer>no</answer>"
        >>> extract_tag_content(text, "answer")
        ['yes', 'no']
    """
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, text, re.DOTALL)
    return matches

# Example usage:
if __name__ == "__main__":
    text = "hello muahaha <answer>yes</answer> and <response>maybe</response> <answer>no</answer>"
    
    # Extract answer tags
    answers = extract_tag_content(text, "answer")
    print("Answers:", answers)  # ['yes', 'no']
    
    # Extract response tags
    responses = extract_tag_content(text, "response")
    print("Responses:", responses)  # ['maybe']