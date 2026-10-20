#!/usr/bin/env python3
"""kokoro_say.py — text -> wav via local Kokoro-82M ONNX (CPU).

Used as a Hermes `tts.providers.<name>: {type: command}` provider:
    kokoro_say.py --text-file <in> --out <out.wav> [--voice af_heart] [--speed 1.0]
Text is passed via file, never over the network. Model lives next to this script.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
LANG_BY_PREFIX = {
    "a": "en-us", "b": "en-gb", "e": "es", "f": "fr-fr", "h": "hi",
    "i": "it", "j": "ja", "p": "pt-br", "z": "zh",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--text-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--model", default="", choices=["", "fp32", "int8"],
                    help="force fp32 or int8 (default: fp32 if present, else int8)")
    args = ap.parse_args()

    if args.text is not None:
        text = args.text
    else:
        text = Path(args.text_file).read_text(encoding="utf-8")
    text = text.strip()
    if not text:
        print(f"no text to speak (input was empty: {args.text_file or '--text'})", file=sys.stderr)
        return 1

    variant = args.model or ("fp32" if (BASE / "kokoro-v1.0.onnx").exists() else "int8")
    model = BASE / ("kokoro-v1.0.onnx" if variant == "fp32" else "kokoro-v1.0.int8.onnx")
    voices = BASE / "voices-v1.0.bin"
    for f in (model, voices):
        if not f.exists():
            print(f"missing {f} — download it (verify SHA256, see README) and place it next to this script", file=sys.stderr)
            return 1

    import soundfile as sf
    from kokoro_onnx import Kokoro

    kokoro = Kokoro(model_path=str(model), voices_path=str(voices))
    lang = LANG_BY_PREFIX.get(args.voice[:1].lower(), "en-us")
    if lang == "zh":
        # espeak backend in kokoro-onnx has no zh; use misaki zh G2P -> phonemes
        from misaki.zh import ZHG2P
        result = ZHG2P()(text)
        phonemes = result[0] if isinstance(result, tuple) else result  # misaki returns (phonemes, tokens); older/newer may differ
        audio, sample_rate = kokoro.create(phonemes, voice=args.voice, speed=args.speed, lang="zh", is_phonemes=True)
    else:
        audio, sample_rate = kokoro.create(text, voice=args.voice, speed=args.speed, lang=lang)
    sf.write(args.out, audio, sample_rate)
    print(f"wrote {args.out} sr={sample_rate} voice={args.voice} model={variant}")
    return 0


if __name__ == "__main__":
    sys.exit(main())