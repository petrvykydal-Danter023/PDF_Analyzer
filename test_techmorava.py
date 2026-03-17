from extraction_logic import heuristic_extract
import json

def mock_word(text, x, y, w, h):
    return {"text": text, "x": x, "y": y, "w": w, "h": h, "cy": y + h/2}

def mock_line(text, words_data):
    words = [mock_word(w[0], w[1], w[2], w[3], w[4]) for w in words_data]
    return {"text_united": text, "words": words, "cy": sum(w['cy'] for w in words)/len(words)}

# TechMorava Example
lines = [
    # Primary Invoice No. vs Symbol
    mock_line("DAŇOVÝ DOKLAD FAKTURA č. 15786435", [("DAŇOVÝ", 100, 10, 80, 30), ("DOKLAD", 200, 10, 80, 30), ("FAKTURA", 300, 10, 100, 30), ("č.", 410, 10, 20, 30), ("15786435", 440, 10, 100, 30)]),
    mock_line("Dodavatel : TechMorava Systems s.r.o. OFFICIAL", [("Dodavatel", 10, 50, 80, 20), (":", 100, 50, 10, 20), ("TechMorava", 120, 50, 100, 20), ("Systems", 230, 50, 80, 20)]),
    mock_line("| variabilní symbol : 121401724", [("|", 400, 50, 10, 20), ("variabilní", 420, 50, 80, 20), ("symbol", 510, 50, 60, 20), (":", 580, 50, 10, 20), ("121401724", 600, 50, 100, 20)]),
    
    # Multi-label line
    mock_line("Místo určení : Bohemia Consulting Odběratel: Vltava IT Services", [
        ("Místo", 10, 200, 50, 20), ("určení", 70, 200, 60, 20), (":", 140, 200, 10, 20), ("Bohemia", 160, 200, 80, 20),
        ("Odběratel:", 400, 200, 80, 20), ("Vltava", 490, 200, 60, 20), ("IT", 560, 200, 20, 20), ("Services", 590, 200, 80, 20)
    ]),

    # Product with 4 decimals and dimension 18x110mm
    mock_line("1. 3M 9528-18x110mm 1 000.00 ks 0.2600 EUR 260.000 21 IT", [
        ("1.", 10, 400, 20, 20), ("3M", 40, 400, 30, 20), ("9528-18x110mm", 80, 400, 120, 20), ("1", 210, 400, 10, 20), ("000.00", 225, 400, 60, 20),
        ("ks", 300, 400, 20, 20), ("0.2600", 350, 400, 60, 20), ("EUR", 420, 400, 40, 20), ("260.000", 500, 400, 80, 20), ("21", 600, 400, 20, 20)
    ]),
    
    # Large total with multiple values nearby
    mock_line("Fakturace celkem EUR 891.04", [("Fakturace", 10, 600, 100, 20), ("celkem", 120, 600, 80, 20), ("EUR", 210, 600, 60, 20), ("891.04", 300, 600, 100, 20)]),
    mock_line("Celkem 3 872.96 22 316.10", [("Celkem", 10, 650, 80, 20), ("3", 100, 650, 10, 20), ("872.96", 115, 650, 80, 20), ("22", 200, 650, 30, 20), ("316.10", 235, 650, 80, 20)]),
]

img_shape = (1000, 1000, 3)
all_text = " ".join(l['text_united'] for l in lines)

extracted = heuristic_extract(lines, img_shape, all_text)

print(json.dumps(extracted, indent=2, ensure_ascii=False))
