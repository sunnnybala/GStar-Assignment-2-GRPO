import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem


def build_paragraph(text: str, style_name: str = "BodyText") -> Paragraph:
    styles = getSampleStyleSheet()
    return Paragraph(text, styles[style_name])


def build_bullets(items: list[str]) -> ListFlowable:
    styles = getSampleStyleSheet()
    return ListFlowable(
        [ListItem(Paragraph(item, styles["BodyText"])) for item in items],
        bulletType="bullet",
        start="-",
        leftIndent=18,
    )


def generate_report(output_path: str = "gstar_assignment2_report.pdf") -> None:
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading2"]

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    # Known configuration (from repo)
    model_id = "Qwen/Qwen3-1.7B"
    dataset_id = "justinphan3110/Countdown-Tasks-3to4"
    loss_type = "grpo"
    n_grpo_steps = 80
    rollout_batch_size = 128
    group_size = 8
    gradient_accumulation_steps = 32
    clip_range = 0.2
    advantage_eps = 1e-6
    max_completion_length = 256

    # Observed eval summary from recent run (terminal logs)
    final_eval_accuracy_pct = 41.7
    mean_reward = 0.4172
    avg_output_tokens = 230.8

    story = []
    story.append(Paragraph("GRPO for Countdown: Approach and Results", title_style))
    story.append(Spacer(1, 0.3 * inch))
    story.append(build_paragraph(f"Generated on: {datetime.datetime.utcnow().isoformat()}Z"))
    story.append(Spacer(1, 0.2 * inch))

    # Approach
    story.append(Paragraph("Approach", heading_style))
    story.append(build_paragraph(
        "We implement Group-Reward Policy Optimization (GRPO) to fine-tune a causal LM on the Countdown task. "
        "We use vLLM for fast on-policy rollouts, compute sequence-level advantages per group, "
        "and optimize a PPO-style clipped surrogate loss without a KL penalty."
    ))
    story.append(build_bullets([
        "Rollouts: vLLM with the current policy weights injected each step for on-policy sampling.",
        "Advantages: group-normalized (per group of size G), optionally std-normalized (GRPO vs DR-GRPO).",
        "Loss: per-token PPO clipping; sequence-level reduction with fair masking.",
        f"Hyperparameters: steps={n_grpo_steps}, batch={rollout_batch_size}, group_size={group_size}, "
        f"grad_acc={gradient_accumulation_steps}, clip_range={clip_range}, eps={advantage_eps}, max_tokens={max_completion_length}.",
    ]))
    story.append(Spacer(1, 0.15 * inch))

    # Reward design
    story.append(Paragraph("Reward Design", heading_style))
    story.append(build_bullets([
        "Answer extraction: parse the last <answer>...</answer> segment (DOTALL).",
        "Number validation: exact multiset match between used digits and provided numbers (no in-place mutation).",
        "Safe evaluation: tokenize + shunting-yard to postfix, then evaluate with guards (e.g., division by zero).",
        "Scoring: 1.0 if valid and equals target (with tolerance), 0.1 if <answer> present but invalid, else 0.0.",
    ]))
    story.append(Spacer(1, 0.15 * inch))

    # Evaluation
    story.append(Paragraph("Evaluation Methods", heading_style))
    story.append(build_bullets([
        f"Dataset: {dataset_id} (train/test splits).",
        "Zero-shot eval script generates responses and computes exact-match accuracy via the reward function.",
        "Sampling: temperature=1.0, top_p=1.0, stop at </answer>, include stop in output.",
    ]))
    story.append(Spacer(1, 0.15 * inch))

    # Results
    story.append(Paragraph("Results", heading_style))
    story.append(build_bullets([
        f"Final eval accuracy: {final_eval_accuracy_pct:.2f}% (from latest run logs).",
        f"Mean reward: {mean_reward:.4f}; Avg output tokens: {avg_output_tokens:.1f}.",
        "TensorBoard logs and checkpoints persisted to Modal Volume at /root/output.",
    ]))
    story.append(Spacer(1, 0.15 * inch))

    # Infrastructure
    story.append(Paragraph("Infrastructure & Environment", heading_style))
    story.append(build_bullets([
        f"Model: {model_id}",
        "Frameworks: vLLM 0.7.2, Transformers 4.51.3, PyTorch 2.4.1+cu124, CUDA 12.4 base.",
        "Modal: A100-40GB GPU, persistent Volume for outputs, HF token via Secret.",
    ]))
    story.append(Spacer(1, 0.15 * inch))

    # Notes
    story.append(Paragraph("Notes & Future Work", heading_style))
    story.append(build_bullets([
        "Consider persisting eval summaries (JSON) alongside checkpoints for richer reports.",
        "Optionally add KL control or multi-epoch PPO updates for exploration-control tradeoffs.",
        "Tune group size, temperature, and max tokens for accuracy vs. length trade-offs.",
    ]))

    doc.build(story)


if __name__ == "__main__":
    generate_report()


