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
    "a": "en-us", "b": "en-gb", "e": "es-es", "f": "fr-fr", "h": "hi-in",
    "i": "it-it", "j": "ja-jp", "p": "pt-br", "z": "zh",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--text")
    g.add_argument("--text-file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--model", default="", help="force fp32 or int8")
    args = ap.parse_args()

    text = args.text if args.text else Path(args.text_file).read_text(encoding="utf-8").strip()
    if not text:
        print("no text to speak", file=sys.stderr)
        return 1

    variant = args.model or ("fp32" if (BASE / "kokoro-v1.0.onnx").exists() else "int8")
    model = BASE / ("kokoro-v1.0.onnx" if variant == "fp32" else "kokoro-v1.0.int8.onnx")
    voices = BASE / "voices-v1.0.bin"

    import soundfile as sf
    from kokoro_onnx import Kokoro

    kokoro = Kokoro(model_path=str(model), voices_path=str(voices))
    lang = LANG_BY_PREFIX.get(args.voice[:1].lower(), "en-us")
    if lang == "zh":
        # espeak backend in kokoro-onnx has no zh; use misaki zh G2P -> phonemes
        from misaki.zh import ZHG2P
        phonemes, _ = ZHG2P()(text)
        audio, sample_rate = kokoro.create(phonemes, voice=args.voice, speed=args.speed, lang="zh", is_phonemes=True)
    else:
        audio, sample_rate = kokoro.create(text, voice=args.voice, speed=args.speed, lang=lang)
    sf.write(args.out, audio, sample_rate)
    print(f"wrote {args.out} sr={sample_rate} voice={args.voice} model={variant}")
    return 0


if __name__ == "__main__":
    sys.exit(main())