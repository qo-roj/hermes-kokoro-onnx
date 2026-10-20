# hermes-kokoro-onnx

Local, offline text-to-speech for [Hermes Agent](https://hermes-agent.nousresearch.com) on **Linux** — [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) via ONNX Runtime on CPU, wired as a native Hermes `type: command` TTS provider.

Voice-note text never leaves the host: no API key, no cloud voice, no telemetry.

## Why this exists

The community `kokoro-tts` plugin ([brunosouzas/hermes-kokoro-tts](https://github.com/brunosouzas/hermes-kokoro-tts)) brings local Kokoro to Hermes — but only on macOS/Apple Silicon (MLX backend, launchd agent, `~/Library` paths). This repo is its **Linux counterpart**: the same model, ONNX on CPU instead of MLX, using a mechanism that is already core Hermes — the `tts.providers.<name>: {type: command}` provider — so no plugin, no daemon, no extra Python in the Hermes tree.

It is a standalone implementation, not a fork: no code is shared with the macOS plugin. If you run Hermes on a Mac, use theirs; everywhere else, this.

## What you get

- Kokoro-82M v1.0, 54 voices, 9 languages (English US/GB, Spanish, French, Hindi, Italian, Japanese, Portuguese, Mandarin)
- ~2× faster than real-time on 8 CPU cores (13.5 s of audio synthesized in 7.1 s, fp32)
- ~450 MB disk: 311 MB model + 27 MB voices + a small venv
- Works on Hermes ≥ 0.22.0 (command providers are core, not plugin)

## Install

Single user (e.g. `~/.hermes/kokoro`):

```bash
mkdir -p ~/.hermes/kokoro && cd ~/.hermes/kokoro
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python kokoro-onnx soundfile "misaki[zh]"
cp /path/to/hermes-kokoro-onnx/kokoro_say.py .

for f in kokoro-v1.0.onnx voices-v1.0.bin; do
  curl -sL -O "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/$f"
done

# verify against known-good copies (the ones this repo was tested with):
#   kokoro-v1.0.onnx  beb0d1848dee9a49da392cc3df26958d46cfa35d321edf434f52949153f0df3a
#   voices-v1.0.bin  bca610b8308e8d99f32e6fe4197e7ec01679264efed0cac9140fe9c29f1fbf7d
sha256sum kokoro-v1.0.onnx voices-v1.0.bin
```

Notes on the config command: Hermes substitutes the placeholders **shell-quote-aware** (see `render_command_template` in `tools/tts_command_provider.py`), so values containing spaces or metacharacters are quoted — you don't need to add quotes around them. The `{speed}` value must exist though: `speed: 1.0` in the provider config keeps the wrapper's argparse happy. And `uv venv --python 3.12` downloads a managed CPython 3.12 if the host doesn't have one — it doesn't rely on the system interpreter.

Get `kokoro_say.py` from this repo, then configure Hermes:

```bash
hermes config set --force tts.providers.kokoro.type command
hermes config set --force tts.providers.kokoro.command '<your-python> <your-script> --text-file {input_path} --out {output_path} --voice {voice} --speed {speed}'
hermes config set --force tts.providers.kokoro.voice af_heart
hermes config set --force tts.providers.kokoro.speed 1.0     # required: empty {speed} breaks argparse
hermes config set --force tts.providers.kokoro.format wav
hermes config set --force tts.providers.kokoro.voice_compatible true
hermes config set tts.provider kokoro
```

`--force` is needed because `tts.config set` does not recognize the nested `tts.providers.*` key, but the Hermes runtime reads it (verified against `tools/tts_command_provider.py`). Changes activate on the next gateway restart. Rollback: `hermes config set tts.provider edge`.

### Multi-agent hosts (shared install)

If several Hermes agents run on one Linux box as separate OS users, do **not** duplicate the 450 MB per user:

```bash
sudo mkdir -p /srv/kokoro
sudo cp kokoro-v1.0.onnx voices-v1.0.bin kokoro_say.py /srv/kokoro/
sudo uv venv --python /usr/bin/python3 /srv/kokoro/.venv   # system python, NOT a per-user uv CPython
sudo uv pip install --python /srv/kokoro/.venv/bin/python kokoro-onnx soundfile "misaki[zh]"
sudo chown -R root:sharedgroup /srv/kokoro && sudo chmod 2755 /srv/kokoro
```

Every agent's command template then points at the shared venv + script, and each agent sets its **own** `tts.providers.kokoro.voice` in its own `config.yaml` — per-agent voices from one shared model. Read-only (`2755`, not group-write) is deliberate: nobody needs write access to a model directory.

## Chinese (and other languages)

kokoro-onnx's espeak backend refuses `lang=zh` ("language zh is not supported"). The wrapper routes any `zf_*`/`zm_*` voice through misaki's `ZHG2P` and feeds phonemes with `is_phonemes=True` — Mandarin works. Language is inferred from the voice prefix (`a`=en-us, `b`=en-gb, `e`=es, `f`=fr, `h`=hi, `i`=it, `j`=ja, `p`=pt, `z`=zh).

## Findings worth knowing

- **fp32 beats int8 on CPU.** The int8 quantized model was ~30% slower *and* lower fidelity on an 8-core x86-64 box. Don't assume quantized = faster on CPU; benchmark both.
- **kokoro-onnx API drift**: v0.6.1 renamed `KokoroOnnx` → `Kokoro` (same `.create(text, voice, speed, lang) → (audio, sr)`). Check `dir(kokoro_onnx)` before writing code against it.
- Each synthesis is a fresh process: expect ~4 s of ONNX session load per voice note. Sentence-by-sentence streaming (the Mac plugin's approach) would need a persistent server; voice notes don't care.
- Voice quality varies: Kokoro grades its own voices in `VOICES.md` on the model card — `am_santa` is D- (≈9 min of training data), `af_heart`/`af_bella` are A-/A-. Character-matching a rasp may mean accepting a low grade; check the table before casting voices.

## Credits

- [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — the model (Apache-2.0)
- [thewh1teagle/kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx) — ONNX packaging and runtime (MIT)
- [brunosouzas/hermes-kokoro-tts](https://github.com/brunosouzas/hermes-kokoro-tts) — the original macOS Hermes plugin that this is the Linux counterpart of
- [misaki](https://github.com/hexgrad/misaki) — G2P, including Mandarin

## License

MIT — see [LICENSE](LICENSE). Model and voices are Apache-2.0 via Kokoro-82M.