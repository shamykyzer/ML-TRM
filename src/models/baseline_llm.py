import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer


class BaselineLLM(nn.Module):
    """Fine-tuned LLM baseline (GPT-2 / TinyLlama) with LoRA.

    Wraps a HuggingFace causal LM with PEFT LoRA adapters.
    Puzzles are formatted as flat token sequences for causal LM training.
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        lora_r: int = 8,
        lora_alpha: int = 16,
        use_qlora: bool = False,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
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

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def trainable_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def total_param_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())
