from extraction_logic import heuristic_extract
import json

def mock_word(text, x, y, w, h):
    return {"text": text, "x": x, "y": y, "w": w, "h": h, "cy": y + h/2}

def mock_line(text, words_data):
    words = [mock_word(w[0], w[1], w[2], w[3], w[4]) for w in words_data]
    return {"text_united": text, "words": words, "cy": sum(w['cy'] for w in words)/len(words)}

# Apple Invoice - Seller in footer, Difficult Total, VAT in products
lines = [
    mock_line("INVOICE 2025-0457", [("INVOICE", 0, 50, 100, 40), ("2025-0457", 110, 50, 150, 40)]),
    mock_line("Customer Apple Czech s.r.o.", [("Customer", 0, 200, 80, 20), ("Apple", 100, 200, 60, 20), ("Czech", 170, 200, 60, 20), ("s.r.o.", 240, 200, 50, 20)]),
    
    # Product with VAT 21% inside
    mock_line("3 ks © Školení personálu 21% 413.22 CZK 1,239.67 CZK", [
        ("3", 0, 400, 20, 20), ("ks", 30, 400, 30, 20), ("©", 70, 400, 20, 20), ("Školení", 100, 400, 80, 20), 
        ("personálu", 190, 400, 100, 20), ("21%", 300, 400, 40, 20), ("413.22", 400, 400, 80, 20), ("CZK", 490, 400, 50, 20),
        ("1,239.67", 600, 400, 100, 20), ("CZK", 710, 400, 50, 20)
    ]),

    # Footer Seller near IČO
    mock_line("Thank You! Bořivoj Hejsek", [("Thank", 0, 900, 60, 20), ("You!", 70, 900, 50, 20), ("Bořivoj", 150, 900, 80, 20), ("Hejsek", 240, 900, 80, 20)]),
    mock_line("Reg.no. 87654321", [("Reg.no.", 0, 930, 80, 20), ("87654321", 90, 930, 80, 20)]),
    
    # Large price at the bottom (Total)
    mock_line("151,690.00 CZK", [("151,690.00", 600, 850, 120, 30), ("CZK", 730, 850, 60, 30)]),
]

img_shape = (1000, 1000, 3)
all_text = " ".join(l['text_united'] for l in lines)

extracted = heuristic_extract(lines, img_shape, all_text)

print(json.dumps(extracted, indent=2, ensure_ascii=False))
