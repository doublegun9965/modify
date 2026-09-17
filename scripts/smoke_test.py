"""Send one correction request. This is a serving check, not an accuracy benchmark."""
import argparse
import json
import time
import urllib.error
import urllib.request

from common import config, run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--text", default="The qu1ck brown f0x jumps over the lazy d0g.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    cfg = config()
    base = (args.base_url or f"http://127.0.0.1:{cfg['port']}").rstrip("/")
    output = run_dir("smoke")
    payload = {"model": cfg["model_path"], "messages": [{"role": "user", "content":
               "Correct OCR errors in the following text. Preserve its meaning and output only the corrected text.\n\n"
               + args.text}], "max_tokens": args.max_tokens, "temperature": 0}
    record = {"base_url": base, "request": payload, "config": cfg,
              "purpose": "prompted corrected copy; not in-place token editing"}
    start = time.monotonic()
    try:
        request = urllib.request.Request(base + "/v1/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            data = json.load(response)
        record["response"] = data
        content = data["choices"][0]["message"]["content"]
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Server returned no non-empty text")
        print(content)
        print("Serving check passed; inspect correction quality separately.")
    except Exception as exc:
        record["error"] = str(exc)
        if isinstance(exc, urllib.error.HTTPError):
            record["error_body"] = exc.read().decode(errors="replace")
        raise
    finally:
        record["elapsed_seconds"] = time.monotonic() - start
        (output / "result.json").write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved: {output / 'result.json'}")


if __name__ == "__main__":
    main()
