from extraction_logic import heuristic_extract
import json

def mock_word(text, x, y, w, h):
    return {"text": text, "x": x, "y": y, "w": w, "h": h, "cy": y + h/2}

def mock_line(text, words_data):
    words = [mock_word(w[0], w[1], w[2], w[3], w[4]) for w in words_data]
    return {"text_united": text, "words": words, "cy": sum(w['cy'] for w in words)/len(words)}

# Global Tech Solutions - Columnar Layout
# BILL FROM: [Supplier]    BILL TO: [Customer]
# ...
# TOTAL [Price] (in a different line or far right)

lines = [
    # Header with labels on the same line, values might be far
    mock_line("BILL FROM: Global Tech Solutions Ltd.", [("BILL", 0, 50, 40, 20), ("FROM:", 45, 50, 40, 20), ("Global", 100, 50, 60, 20), ("Tech", 170, 50, 50, 20)]),
    mock_line("BILL TO: Sunrise Logistics Inc.", [("BILL", 400, 50, 40, 20), ("TO:", 445, 50, 30, 20), ("Sunrise", 500, 50, 70, 20), ("Logistics", 580, 50, 80, 20)]),
    
    # OCR Error: "Ouantity" instead of "Quantity"
    mock_line("Ouantity Unit Cost Line Total", [("Ouantity", 400, 200, 80, 20), ("Unit", 500, 200, 40, 20), ("Cost", 550, 200, 40, 20), ("Line", 600, 200, 40, 20)]),
    
    # Total far away from label
    mock_line("TOTAL", [("TOTAL", 400, 450, 60, 20)]),
    mock_line("866.39", [("866.39", 600, 450, 70, 20)]), # Far right of "TOTAL"
]

img_shape = (1000, 1000, 3)
all_text = " ".join(l['text_united'] for l in lines)

extracted = heuristic_extract(lines, img_shape, all_text)

print(json.dumps(extracted, indent=2, ensure_ascii=False))
