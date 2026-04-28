from transformers import AutoTokenizer


def build_tokenizer(tokenizer_name: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer