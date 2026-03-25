#!/usr/bin/env python3
from __future__ import annotations

import os
import sys


def main() -> int:
    key = os.environ.get('GEMINI_API_KEY')
    if not key:
        print('GEMINI_API_KEY is not set')
        return 1
    try:
        from google import genai
    except Exception as exc:
        print(f'Failed to import google.genai: {exc}')
        return 2
    try:
        client = genai.Client()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents='Reply with exactly: ok',
        )
        text = getattr(response, 'text', None)
        print(text if text is not None else response)
        return 0
    except Exception as exc:
        print(f'Gemini call failed: {exc}')
        return 3


if __name__ == '__main__':
    raise SystemExit(main())
