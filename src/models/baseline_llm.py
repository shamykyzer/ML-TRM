import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


# Per-task token-id schemas. Fix B remaps the dataset's compact id space onto
# real semantic single-character LLM tokens so the pretrained model uses its
# learned embeddings instead of whatever punctuation happens to sit at vocab
# positions 0..K-1, then projects the LLM's full-vocab logits back down to K
# columns so trainer / eval / distill see logits.shape[-1] == vocab_size.
#
# Sudoku (vocab=11): id 0 pad, 1 blank, 2..10 digits 1..9.
# Maze   (vocab=6 ): id 0 pad, 1 '#' wall, 2 ' ' open, 3 'S' start, 4 'G' goal,
#                   5 'o' path marker (matches data/build_maze_dataset.py CHARSET).
SUDOKU_ID_TO_CHAR = {
    0: ".",   # never queried at input; harmless fallback if it ever is
    1: ".",   # blank
    2: "1", 3: "2", 4: "3", 5: "4", 6: "5",
    7: "6", 8: "7", 9: "8", 10: "9",
}
MAZE_ID_TO_CHAR = {
    0: ".",   # never queried at input; harmless fallback
    1: "#", 2: " ", 3: "S", 4: "G", 5: "o",
}
SUDOKU_VOCAB_SIZE = 11
MAZE_VOCAB_SIZE = 6
VOCAB_MAPS: dict[int, dict[int, str]] = {
    SUDOKU_VOCAB_SIZE: SUDOKU_ID_TO_CHAR,
    MAZE_VOCAB_SIZE:   MAZE_ID_TO_CHAR,
}


class BaselineLLM(nn.Module):
    """Fine-tuned LLM baseline (GPT-2 / TinyLlama / Qwen) with LoRA.

    Wraps a HuggingFace causal LM with PEFT LoRA adapters and remaps a compact
    task-vocabulary onto the LLM's own single-character tokens (Fix B). Inputs
    and labels enter as task ids; outputs leave as task-id-space logits, so the
    trainer / eval / distillation paths see logits.shape[-1] == vocab_size
    regardless of which task or LLM backbone is used. Pass ``vocab_size=11``
    for sudoku (default) or ``vocab_size=6`` for maze.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        lora_r: int = 8,
        lora_alpha: int = 16,
        use_qlora: bool = False,
        use_gradient_checkpointing: bool = False,
        vocab_size: int = SUDOKU_VOCAB_SIZE,
    ):
        super().__init__()
        # Fail fast on unknown task vocabularies. The Fix B remap depends on a
        # per-task id->char table; without one we'd silently feed task tokens
        # through whichever existing buffer matches len(id_to_char), producing
        # garbage. Adding a new task = add an entry to VOCAB_MAPS above.
        if vocab_size not in VOCAB_MAPS:
            raise ValueError(
                f"BaselineLLM has no Fix-B vocab map for vocab_size={vocab_size}; "
                f"known: {sorted(VOCAB_MAPS)}. Add an entry to VOCAB_MAPS."
            )
        id_to_char = VOCAB_MAPS[vocab_size]
        self.vocab_size = vocab_size
        from peft import LoraConfig, TaskType, get_peft_model

        load_kwargs = {}
        if use_qlora:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

        base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        if use_qlora:
            from peft import prepare_model_for_kbit_training

            # prepare_model_for_kbit_training enables gradient checkpointing and
            # calls enable_input_require_grads so LoRA gradients flow correctly.
            base_model = prepare_model_for_kbit_training(
                base_model,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )

        # Determine target modules based on model architecture.
        # DeepSeek-R1-Distill-Qwen inherits Qwen's module names; -Llama variant
        # inherits Llama's. Match DeepSeek first to avoid a generic fallback.
        name_lower = model_name.lower()
        if "deepseek" in name_lower and "llama" in name_lower:
            target_modules = ["q_proj", "v_proj"]
        elif "deepseek" in name_lower:
            target_modules = ["q_proj", "k_proj", "v_proj"]
        elif "gpt2" in name_lower:
            target_modules = ["c_attn", "c_proj"]
        elif "qwen" in name_lower:
            target_modules = ["q_proj", "k_proj", "v_proj"]
        elif "smollm" in name_lower:
            target_modules = ["q_proj", "k_proj", "v_proj"]
        elif "llama" in name_lower:
            target_modules = ["q_proj", "v_proj"]
        else:
            target_modules = ["q_proj", "v_proj"]

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=target_modules,
        )
        self.model = get_peft_model(base_model, peft_config)

        # Non-QLoRA path: enable gradient checkpointing AFTER get_peft_model so
        # PEFT's module-rewrapping can't silently reset the flag. The QLoRA path
        # above already handled this via prepare_model_for_kbit_training.
        # enable_input_require_grads is required because the frozen base model's
        # input embeddings don't require grad, which would break backprop
        # through the checkpoint boundary.
        if use_gradient_checkpointing and not use_qlora:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Build task-id -> LLM-token-id lookup. We require each character to
        # encode to exactly one token; if the tokenizer splits the maze space
        # ' ' or any digit into multiple subwords, abort loudly rather than
        # silently truncate. (GPT-2 byte-level BPE encodes bare " " as 1 token
        # at id 220; Qwen's tiktoken-style BPE does the same.)
        ids = []
        for i in range(vocab_size):
            ch = id_to_char[i]
            tok_ids = self.tokenizer.encode(ch, add_special_tokens=False)
            if len(tok_ids) != 1:
                raise ValueError(
                    f"Tokenizer for {model_name!r} encoded {ch!r} as "
                    f"{len(tok_ids)} tokens ({tok_ids}); BaselineLLM Fix B "
                    f"requires every char in the task vocab to be 1 token."
                )
            ids.append(tok_ids[0])
        self.register_buffer(
            "task_to_llm",
            torch.tensor(ids, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        # Remap task ids -> LLM token ids before the embedding lookup.
        llm_inputs = self.task_to_llm[input_ids]

        # Forward without HF's internal labels handling; we recompute loss in
        # vocab_size-dim task space below so the gradient signal is "predict
        # the right task token", not "avoid the other ~50K LLM-vocab tokens".
        out = self.model(
            input_ids=llm_inputs,
            attention_mask=attention_mask,
        )

        # Project the LLM's full-vocab logits down to the task's vocab_size
        # columns (in task-id order). Trainer / eval / distillation now see
        # logits.shape[-1] == self.vocab_size.
        out.logits = out.logits.index_select(-1, self.task_to_llm)

        # Re-derive loss in the task-vocab space using the standard HF causal-LM
        # shift (logits[i] predicts token i+1). Labels are task ids with the
        # trainer's -100 ignore sentinel preserved.
        if labels is not None:
            shifted_logits = out.logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()
            out.loss = F.cross_entropy(
                shifted_logits.view(-1, shifted_logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=-100,
            )
        return out

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
