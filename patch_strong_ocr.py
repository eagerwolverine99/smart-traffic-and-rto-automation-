"""
Patch script — adds the missing LETTER_CONFUSIONS, INDIAN_STATE_CODES,
and correct_state_code to your strong_ocr.py.

Run ONCE:
    python patch_strong_ocr.py
"""

import os
import shutil

FILE = "strong_ocr.py"

PATCH = '''

# ============================================================
# PATCH: Added by patch_strong_ocr.py for main_v6 compatibility
# ============================================================

# Letter-to-letter OCR confusions common on license plates
LETTER_CONFUSIONS = {
    'X': 'K',   # X misread for K
    'H': 'M',   # H misread for M
    'N': 'M',   # N misread for M
    'V': 'Y',
    'U': 'V',
    'C': 'G',
    'E': 'F',
    'F': 'E',
}

# All valid Indian RTO state/UT codes
INDIAN_STATE_CODES = {
    'AP', 'AR', 'AS', 'BR', 'CG', 'DL', 'GA', 'GJ', 'HR', 'HP',
    'JK', 'JH', 'KA', 'KL', 'LD', 'MP', 'MH', 'MN', 'ML', 'MZ',
    'NL', 'OD', 'PB', 'RJ', 'SK', 'TN', 'TS', 'TR', 'UP', 'UK',
    'WB', 'CH', 'DD', 'DN', 'LA', 'PY', 'AN',
}


def correct_state_code(first_two):
    """
    Snap a 2-char state-code reading to the nearest valid Indian state code.
    Handles common OCR confusions:
      - XA -> KA (Karnataka)
      - XL -> KL (Kerala)
      - HP, MP, etc. return unchanged
    """
    if first_two in INDIAN_STATE_CODES:
        return first_two

    candidates = [first_two]
    for i in range(2):
        ch = first_two[i]
        fix = LETTER_CONFUSIONS.get(ch)
        if fix:
            candidates.append(first_two[:i] + fix + first_two[i+1:])

    # Digit-to-letter substitutions for state-code positions
    digit_to_letter = {
        '0': 'O', '1': 'I', '2': 'Z',
        '4': 'A', '5': 'S', '6': 'G',
        '7': 'T', '8': 'B', '9': 'P',
    }
    for i in range(2):
        ch = first_two[i]
        if ch.isdigit():
            fix = digit_to_letter.get(ch)
            if fix:
                candidates.append(first_two[:i] + fix + first_two[i+1:])

    for c in candidates:
        if c in INDIAN_STATE_CODES:
            return c

    def hamming(a, b):
        return sum(1 for x, y in zip(a, b) if x != y)

    best = min(INDIAN_STATE_CODES, key=lambda s: hamming(first_two, s))
    return best if hamming(first_two, best) <= 1 else first_two
'''


def main():
    if not os.path.exists(FILE):
        print(f"[ERROR] {FILE} not found in current folder.")
        print("        Make sure you're in C:\\Users\\saini\\Downloads\\anpr\\")
        return

    # Read existing file
    with open(FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    # Back up first
    backup = FILE + ".bak"
    shutil.copy(FILE, backup)
    print(f"[OK] Backed up existing file to: {backup}")

    # Check if already patched
    if 'LETTER_CONFUSIONS' in content and 'correct_state_code' in content:
        print("[OK] File already contains the required symbols. Nothing to do.")
        return

    # Append the patch
    with open(FILE, 'a', encoding='utf-8') as f:
        f.write(PATCH)

    new_size = os.path.getsize(FILE)
    print(f"[OK] Patched {FILE} (new size: {new_size} bytes)")
    print("[OK] Now verify with:")
    print("     python -c \"from strong_ocr import LETTER_CONFUSIONS; print('works')\"")


if __name__ == "__main__":
    main()
