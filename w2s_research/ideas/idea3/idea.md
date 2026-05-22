You could try to RL optimize a model to produce text which satisfies the following reward signals:
It has high log-prob likelihood under the poisoned system prompt
An LLM judge is fooled by it
Basically, we could take a qwen, gemma or olmo and optimize it to produce text like this on some small held-out set of Alpaca. We then run it on the rest of alpaca to produce the poisoned data. You can use ARGO for this:

ARGO optimizes system prompts by RL-training a small "proposer" LM (LoRA on top of a frozen base, e.g. Llama-3.2-3B-Instruct) to generate natural-language system prompts that minimize a black-box scorer signal
   (typically reverse/forward-KL or cross-entropy between a target model under the candidate prompt and a teacher). Each outer step it (1) samples a group of G candidate prompts from the current LoRA policy via
  temperature/top-p sampling off a fixed prompt-engineer meta-prompt; (2) scores each candidate with the configured Scorer — preferring a batched score_candidates_grid path when available — to get per-candidate
  mean KL, and assembles a reward of -mean_KL - length_coef * (completion_len / max_prompt_tokens), with invalid/empty proposals getting a -10 sentinel; (3) forms a leave-one-out advantage adv_i = G/(G-1) · (r_i
   - mean(r)) (deliberately not std-normalized, to avoid GRPO's small-spread gradient blow-ups); (4) computes a GRPO-style policy-gradient loss on completion-masked log-probs, plus a Schulman-k3 per-token KL
  penalty to a reference policy (the same proposer with LoRA disabled) and a per-token entropy bonus, i.e. pg_loss + kl_coef·KL(π‖π_ref) - entropy_coef·H(π); (5) takes one AdamW step on the LoRA parameters with
  grad-norm clipping (no PPO clip, single update per rollout group); and (6) tracks the best-reward prompt seen and periodically runs a clean scorer eval on it. Final artifact is best_prompt.txt plus
  argo_state.json and an iterations.jsonl trace.

