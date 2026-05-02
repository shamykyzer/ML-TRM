import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForCausalLM, AutoTokenizer


# Sudoku stored-token schema (see src/data/sudoku_dataset.py module docstring):
#   id 0  = pad / ignore (only ever appears in labels, remapped to -100 by trainer)
#   id 1  = blank cell
#   id 2..10 = digits 1..9
# We map each sudoku id to its semantic single-character GPT-2-style token so the
# pretrained LLM uses its real digit embeddings instead of the punctuation tokens
# that happen to live at GPT-2 vocab positions 0-10. This stops the model from
# collapsing onto a "predict any token in [0,11)" digit-prior shortcut and
# forces it to actually engage with the constraint-satisfaction problem.
SUDOKU_ID_TO_CHAR = {
    0: ".",   # never queried at input; harmless fallback if it ever is
    1: ".",   # blank
    2: "1", 3: "2", 4: "3", 5: "4", 6: "5",
    7: "6", 8: "7", 9: "8", 10: "9",
}

# Maze stored-token schema (see src/data/maze_dataset.py module docstring —
# CHARSET = '# SGo'):
#   id 0  = pad / ignore (only ever appears in labels, remapped to -100 by trainer)
#   id 1  = # (wall)
#   id 2  = ' ' (open cell)  — mapped to '_' for single-token GPT-2 compatibility
#                              (BPE handles ' ' via the Ġ-prefix mechanism, which
#                               makes the bare " " encode awkwardly; '_' is a
#                               clean single-token semantic stand-in)
#   id 3  = S (start)
#   id 4  = G (goal)
#   id 5  = o (path marker — the solution the model must output)
# Same Fix-B principle as sudoku: route each maze cell type to a semantic
# single-character token in the LLM's vocab so the LoRA learns "wall" via the
# GPT-2 '#' embedding instead of an arbitrary punctuation slot. Without this,
# maze training silently reused the sudoku digit map (cell type S -> token '1',
# etc.), wasting capacity and giving the LLM no semantic anchoring for maze
# structure.
MAZE_ID_TO_CHAR = {
    0: ".",   # pad / ignore
    1: "#",   # wall
    2: "_",   # open cell
    3: "S",   # start
    4: "G",   # goal
    5: "o",   # path marker
}

TASK_VOCAB = {
    "sudoku": {"size": 11, "id_to_char": SUDOKU_ID_TO_CHAR},
    "maze":   {"size": 6,  "id_to_char": MAZE_ID_TO_CHAR},
}

SUDOKU_VOCAB_SIZE = TASK_VOCAB["sudoku"]["size"]


class BaselineLLM(nn.Module):
    """Fine-tuned LLM baseline (GPT-2 / TinyLlama / Qwen) with LoRA.

    Wraps a HuggingFace causal LM with PEFT LoRA adapters and remaps a compact
    task-vocabulary onto the LLM's own single-character tokens (Fix B). Inputs
    and labels enter as task ids; outputs leave as task-id-space logits, so the
    trainer / eval / distillation paths see logits.shape[-1] == vocab_size
    regardless of which task or LLM backbone is used.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        lora_r: int = 8,
        lora_alpha: int = 16,
        use_qlora: bool = False,
        use_gradient_checkpointing: bool = False,
        task: str = "sudoku",
    ):
        super().__init__()
        from peft import LoraConfig, TaskType, get_peft_model

        if task not in TASK_VOCAB:
            raise ValueError(
                f"Unknown task {task!r}; expected one of {sorted(TASK_VOCAB)}."
            )
        self.task = task

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

            base_model = prepare_model_for_kbit_training(
                base_model,
                use_gradient_checkpointing=use_gradient_checkpointing,
            )

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

        if use_gradient_checkpointing and not use_qlora:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        spec = TASK_VOCAB[task]
        vocab_size = spec["size"]
        id_to_char = spec["id_to_char"]
        ids = []
        for i in range(vocab_size):
            ch = id_to_char[i]
            tok_ids = self.tokenizer.encode(ch, add_special_tokens=False)
            if len(tok_ids) != 1:
                raise ValueError(
                    f"Tokenizer for {model_name!r} encoded {ch!r} as "
                    f"{len(tok_ids)} tokens ({tok_ids}); BaselineLLM Fix B "
                    f"requires every {task!r} vocab entry to map to a "
                    f"single LLM token."
                )
            ids.append(tok_ids[0])
        self.register_buffer(
            "vocab_to_llm",
            torch.tensor(ids, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        llm_inputs = self.vocab_to_llm[input_ids]

        out = self.model(
            input_ids=llm_inputs,
            attention_mask=attention_mask,
        )

        out.logits = out.logits.index_select(-1, self.vocab_to_llm)

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
