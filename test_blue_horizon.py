from extraction_logic import heuristic_extract
import json

def mock_word(text, x, y, w, h):
    return {"text": text, "x": x, "y": y, "w": w, "h": h, "cy": y + h/2}

def mock_line(text, words_data):
    words = [mock_word(w[0], w[1], w[2], w[3], w[4]) for w in words_data]
    return {"text_united": text, "words": words, "cy": sum(w['cy'] for w in words)/len(words)}

# Blue Horizon Technologies LLC - Header without label
# SHIP TO label
# "Invoice No. 241914"
# "$1,200.00" comma separator
# "2000 x M1.2..." product format

lines = [
    mock_line("Blue Horizon Technologies LLC", [("Blue", 0, 10, 50, 20), ("Horizon", 55, 10, 80, 20), ("Technologies", 140, 10, 100, 20), ("LLC", 245, 10, 40, 20)]),
    mock_line("SHIP TO: Delta Office Technology s.r.o.", [("SHIP", 0, 100, 50, 20), ("TO:", 55, 100, 30, 20), ("Delta", 100, 100, 60, 20), ("Office", 165, 100, 60, 20)]),
    mock_line("Invoice No. 241914", [("Invoice", 400, 100, 60, 20), ("No.", 465, 100, 30, 20), ("241914", 500, 100, 60, 20)]),
    
    # Total with comma
    mock_line("Total: $1,205.00", [("Total:", 400, 500, 60, 20), ("$1,205.00", 470, 500, 100, 20)]),
    
    # Product row "Qty x Name..."
    mock_line("2000 x M1.2-0.25 x 2.5mm Stainless Steel $0.60 $1,200.00", [
        ("2000", 0, 300, 50, 20), 
        ("x", 55, 300, 20, 20), 
        ("M1.2-0.25", 80, 300, 100, 20), 
        ("Stainless", 190, 300, 80, 20),
        ("$0.60", 400, 300, 60, 20),
        ("$1,200.00", 470, 300, 100, 20)
    ]),
]

img_shape = (1000, 1000, 3)
all_text = " ".join(l['text_united'] for l in lines)

extracted = heuristic_extract(lines, img_shape, all_text)

print(json.dumps(extracted, indent=2, ensure_ascii=False))
