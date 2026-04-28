from transformers import GPT2Config, GPT2LMHeadModel


def build_model(cfg, tokenizer):
    model_config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=cfg.block_size,
        n_embd=cfg.n_embd,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        pad_token_id=tokenizer.pad_token_id,
    )

    model = GPT2LMHeadModel(model_config)
    return model