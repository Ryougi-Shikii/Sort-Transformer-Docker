"""
app.py — Gradio demo for the SortTransformer
Loads the self-contained checkpoint (vocab + model config + weights)
and runs greedy / beam-search decoding on user input.
"""

import torch
import gradio as gr

from model     import Transformer
from dataset   import Vocabulary
from inference import greedy_decode, beam_search

# ── CONFIG ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "sort_model.pt"
DEVICE          = "cpu"          # HF free-tier has no GPU
MAX_LEN         = 50
BEAM_SIZE       = 4

# ── LOAD CHECKPOINT ───────────────────────────────────────────────────────────
checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

src_vocab = Vocabulary()
src_vocab.token2idx = checkpoint["src_token2idx"]
src_vocab.idx2token = {int(k): v for k, v in checkpoint["src_idx2token"].items()}

tgt_vocab = Vocabulary()
tgt_vocab.token2idx = checkpoint["tgt_token2idx"]
tgt_vocab.idx2token = {int(k): v for k, v in checkpoint["tgt_idx2token"].items()}

model = Transformer(**checkpoint["model_config"]).to(DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print("Model loaded successfully.")
print(f"Vocab size: {len(src_vocab)}")

# ── INFERENCE ─────────────────────────────────────────────────────────────────
SPECIAL = {Vocabulary.PAD, Vocabulary.SOS, Vocabulary.EOS, Vocabulary.UNK}

def decode_ids(ids):
    return "".join(
        tgt_vocab.idx2token[i] for i in ids if i not in SPECIAL
    )

def predict(word: str):
    word = word.strip().lower()

    if not word:
        return "", "", "", ""

    # Filter out characters not in vocab
    unknown = [c for c in word if c not in src_vocab.token2idx]
    if unknown:
        msg = f"Unknown characters: {list(set(unknown))}. Only a–z are supported."
        return msg, "", "", ""

    src = torch.tensor(
        [src_vocab.encode(list(word))], dtype=torch.long, device=DEVICE
    )

    greedy_ids = greedy_decode(
        model, src, Vocabulary.SOS, Vocabulary.EOS,
        max_len=MAX_LEN, device=DEVICE
    )
    beam_ids = beam_search(
        model, src, Vocabulary.SOS, Vocabulary.EOS,
        beam_size=BEAM_SIZE, max_len=MAX_LEN, device=DEVICE
    )

    greedy_out = decode_ids(greedy_ids)
    beam_out   = decode_ids(beam_ids)
    expected   = "".join(sorted(word))

    greedy_status = "✓ Correct" if greedy_out == expected else "✗ Wrong"
    beam_status   = "✓ Correct" if beam_out   == expected else "✗ Wrong"

    return greedy_out, greedy_status, beam_out, beam_status

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="SortTransformer") as demo:
    gr.Markdown(
        """
        # 🔤 SortTransformer
        A from-scratch Transformer trained to sort characters alphabetically.
        Enter any lowercase word and watch it sort letter by letter.

        **Model:** 3-layer Encoder-Decoder · d_model=128 · 4 heads · trained on 10k random sequences
        """
    )

    with gr.Row():
        inp = gr.Textbox(
            label="Input word",
            placeholder="e.g. transformer",
            max_lines=1,
        )
        btn = gr.Button("Sort", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Greedy Decode")
            greedy_out    = gr.Textbox(label="Output")
            greedy_status = gr.Textbox(label="Verdict")
        with gr.Column():
            gr.Markdown("### Beam Search (beam=4)")
            beam_out    = gr.Textbox(label="Output")
            beam_status = gr.Textbox(label="Verdict")

    gr.Examples(
        examples=["hello", "transformer", "python", "abcde", "zyxwvu"],
        inputs=inp,
    )

    btn.click(
        fn=predict,
        inputs=inp,
        outputs=[greedy_out, greedy_status, beam_out, beam_status],
    )
    inp.submit(
        fn=predict,
        inputs=inp,
        outputs=[greedy_out, greedy_status, beam_out, beam_status],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
