from tqdm import tqdm

VOCABULARY = [0, 1000, # Special token (since token dublicates are not allowed, we don't need special separator token)
              1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, # Unspittable numbers 1-19
              20, 30, 40, 50, 60, 70, 80, 90, # Unspittable numbers 10-th powers
              100, 200, 300, 400, 500, 600, 700, 800, 900] # Unspittable numbers 100-900
REVERSE_VOCABULARY = {v: i for i, v in enumerate(VOCABULARY)}

def tokens_to_number(tokens: list[int]) -> str:
    """
    Convert a list of tokens to a number string.
    """
    number = 0
    prev_token = REVERSE_VOCABULARY[1000]
    for token in tokens:
        if token == prev_token:
            prev_token = token
            continue
        symbol = VOCABULARY[token]
        if symbol == 1000:
            number *= 1000
        else:
            number += symbol
        prev_token = token
    return str(number)

def number_to_tokens(number: str) -> list[int]:
    """
    Convert a number string to a list of tokens.
    """
    assert number.isdigit(), "Input must be a digit string"
    int_number = int(number)
    assert 1_000 <= int_number < 1_000_000, "Number must be between 1000 and 999999"
    
    major_part = int_number // 1000
    minor_part = int_number % 1000

    tokens = []

    if major_part >= 100:
        tokens.append(REVERSE_VOCABULARY[major_part // 100 * 100])
        major_part %= 100
    if major_part >= 20:
        tokens.append(REVERSE_VOCABULARY[major_part // 10 * 10])
        major_part %= 10
    if major_part > 0:
        tokens.append(REVERSE_VOCABULARY[major_part])
    tokens.append(REVERSE_VOCABULARY[1000]) # Special token for 1000
    if minor_part >= 100:
        tokens.append(REVERSE_VOCABULARY[minor_part // 100 * 100])
        minor_part %= 100
    if minor_part >= 20:
        tokens.append(REVERSE_VOCABULARY[minor_part // 10 * 10])
        minor_part %= 10
    if minor_part > 0:
        tokens.append(REVERSE_VOCABULARY[minor_part])

    return tokens

def check_token_seq_possibility(tokens: list[int]) -> bool:
    """
    Check if a token sequence is possible.
    """

    def next_largest_available_token(token: int) -> int:
        """
        Get the next largest available token.
        """
        symbol = VOCABULARY[token]
        if symbol == 0:
            return REVERSE_VOCABULARY[900] # All tokens are available after blank
        if symbol < 20:
            return REVERSE_VOCABULARY[1000] # Only EOS and 1000 are available after 1-19
        if symbol < 100:
            return REVERSE_VOCABULARY[9] # Only 1-9, 1000 and EOS are available after 10-90
        if symbol < 1000:
            return REVERSE_VOCABULARY[90] # Only 1-90, 1000 and EOS are available after 100-900
        if symbol == 1000:
            return REVERSE_VOCABULARY[900] # All tokens are available after 1000

    prev_token = REVERSE_VOCABULARY[1000]
    for token in tokens:
        if token != prev_token and token > next_largest_available_token(prev_token):
            return False
        prev_token = token

    return True

def numbers_to_tokens(numbers: list[str]) -> list[list[int]]:
    """
    Convert a list of numbers to a list of token sequences.
    """
    return [number_to_tokens(number) for number in numbers]

def tokens_to_numbers(tokens: list[list[int]]) -> list[str]:
    """
    Convert a list of token sequences to a list of numbers.
    """
    return [tokens_to_number(token_seq) for token_seq in tokens]

if __name__ == "__main__":
    for i in tqdm(range(1000, 1000000)):
        number = str(i)
        tokens = number_to_tokens(number)
        assert check_token_seq_possibility(tokens), f"Token sequence is not possible for {number}"
        assert number == tokens_to_number(tokens), f"Tokens do not match for {number}"
        assert tokens == number_to_tokens(tokens_to_number(tokens)), f"Tokens do not match for {number}"
        assert tokens == number_to_tokens(number), f"Tokens do not match for {number}"