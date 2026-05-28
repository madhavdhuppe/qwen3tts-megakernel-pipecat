class ConfigMapper:

    @staticmethod
    def map(cfg):

        return {
            "hidden_size": cfg.hidden_size,
            "num_layers": cfg.num_hidden_layers,
            "num_heads": cfg.num_attention_heads,
            "kv_heads": cfg.num_key_value_heads,
            "vocab_size": cfg.vocab_size
        }