from extraction_logic import heuristic_extract
import json

# Mocking the lines structure from user's OCR text
# "DAŇOVÝ DOKLAD FAKTURA č. 15786435"
# "Dodavatel : TechMorava Systems s.r.o. OFFICIAL | variabilní symbol : 121401724"
# ...

def mock_word(text, x, y, w, h):
    return {"text": text, "x": x, "y": y, "w": w, "h": h, "cy": y + h/2}

def mock_line(text, words_data):
    words = [mock_word(w[0], w[1], w[2], w[3], w[4]) for w in words_data]
    return {"text_united": text, "words": words, "cy": sum(w['cy'] for w in words)/len(words)}

lines = [
    mock_line("DAŇOVÝ DOKLAD FAKTURA č. 15786435", [("DAŇOVÝ", 0,0,100,20), ("DOKLAD", 110,0,100,20), ("FAKTURA", 220,0,100,20), ("č.", 330,0,20,20), ("15786435", 360,0,100,20)]),
    mock_line("Dodavatel : TechMorava Systems s.r.o. OFFICIAL | variabilní symbol : 121401724", [("Dodavatel", 0,30,100,20), (":", 110,30,10,20), ("TechMorava", 130,30,100,20), ("Systems", 240,30,100,20), ("s.r.o.", 350,30,50,20)]),
    mock_line("Odběratel: Vltava IT Services s.r.o.", [("Odběratel:", 0,100,100,20), ("Vltava", 110,100,100,20), ("IT", 220,100,50,20), ("Services", 280,100,100,20), ("s.r.o.", 390,100,50,20)]),
    mock_line("IČ : 29481736 DIČ : CZ29481736", [("IČ", 0,130,30,20), (":", 40,130,10,20), ("29481736", 60,130,80,20), ("DIČ", 150,130,40,20), (":", 200,130,10,20), ("CZ29481736", 220,130,100,20)]),
    mock_line("Fakturace celkem EUR 891.04", [("Fakturace", 0,500,100,20), ("celkem", 110,500,80,20), ("EUR", 200,500,40,20), ("891.04", 250,500,80,20)]),
    # Product line
    mock_line("1. 3M 9528-18x110mm 1 000.00 ks 0.2600 EUR 260.000 21 IT", [("1.", 0,300,20,20), ("3M", 30,300,50,20), ("9528-18x110mm", 90,300,150,20), ("1 000.00", 250,300,80,20), ("ks", 340,300,30,20), ("0.2600", 380,300,60,20), ("EUR", 450,300,40,20), ("260.000", 500,300,80,20)]),
]

img_shape = (2000, 1500, 3)
all_text = " ".join(l['text_united'] for l in lines)

extracted = heuristic_extract(lines, img_shape, all_text)

print(json.dumps(extracted, indent=2, ensure_ascii=False))
