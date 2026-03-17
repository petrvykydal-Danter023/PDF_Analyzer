import pytesseract
from pytesseract import Output
import re
import unicodedata
import requests

from template_engine import (
    TemplateStore,
    detect_kv_pairs,
    detect_table_region,
    detect_table_rows_in_region,
    generate_template,
    apply_template,
)

pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'
store = TemplateStore()


def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def classify_doc(text):
    text_lower = remove_accents(text.lower())
    if "faktura" in text_lower or "danovy doklad" in text_lower or "invoice" in text_lower:
        return "FAKTURA"
    elif "nabidka" in text_lower or "cenova kalkulace" in text_lower or "quote" in text_lower or "estimate" in text_lower:
        return "NABÍDKA"
    else:
        return "EMAIL/OSTATNÍ"


def validate_ico_via_ares(ico):
    """Ověří IČO přes ARES API a vrátí data firmy nebo None."""
    try:
        url = f"https://ares.gov.cz/ekonomicke-subjekty-v-be/rest/ekonomicke-subjekty/{ico}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            sidlo = data.get("sidlo", {})
            return {
                "nazev": data.get("obchodniJmeno", ""),
                "ico": data.get("ico", ""),
                "dic": data.get("dic", ""),
                "adresa": sidlo.get("textovaAdresa", ""),
                "psc": str(sidlo.get("psc", "")),
            }
    except Exception:
        pass
    return None


def build_lines_from_ocr(d):
    """Sestaví Y-osové řádky z Tesseract výstupu."""
    words = []
    for i in range(len(d['text'])):
        text = str(d['text'][i]).strip()
        if not text:
            continue
        left, top, width, height = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        center_y = top + height / 2.0
        words.append({"text": text, "x": left, "y": top, "w": width, "h": height, "cy": center_y})

    words.sort(key=lambda w: w['cy'])
    lines = []
    for w in words:
        if lines and abs(lines[-1]['cy'] - w['cy']) < 15:
            lines[-1]['words'].append(w)
            n = len(lines[-1]['words'])
            lines[-1]['cy'] = (lines[-1]['cy'] * (n - 1) + w['cy']) / n
        else:
            lines.append({'cy': w['cy'], 'words': [w]})

    for line in lines:
        line['words'].sort(key=lambda w: w['x'])
        line['text_united'] = " ".join([w['text'] for w in line['words']])

    return lines


def find_ico_in_lines(lines):
    """Pokusí se najít IČO přímo z textu (8místné číslo za kotvou IČ/IČO/Reg.no)."""
    for line in lines:
        lt = line['text_united'].lower()
        if re.search(r'\b(ic|ico|reg\.?\s*no\.?)\b', lt):
            for w in line['words']:
                if re.match(r'^\d{8}$', w['text']):
                    return w['text']
    return None


def heuristic_extract(lines, img_shape, all_text):
    """Fallback: heuristická extrakce (náš původní algoritmus)."""
    img_h, img_w = img_shape[:2]
    data = {
        "metadata": {"typ_dokumentu": classify_doc(all_text), "stranky": 1},
        "prodavajici": {},
        "odberatel": {},
        "faktura": {},
        "produkty": [],
        "celkem": {}
    }

    in_customer_block = 0

    for i, line in enumerate(lines):
        lt = line['text_united']
        lt_lower = lt.lower()
        lt_lower_na = remove_accents(lt_lower)

        if re.search(r'\b(odberatel|customer|prijemce|ship to|sold to)\b', lt_lower_na):
            in_customer_block = 8
            continue

        if in_customer_block > 0:
            if re.search(r'\b(ic|ico|reg\.?\s*no\.?)\b', lt_lower):
                for w in line['words']:
                    if re.match(r'^\d{8}$', w['text']):
                        data["odberatel"]["ico"] = {"hodnota": w['text'], "box": [w['x'], w['y'], w['w'], w['h']]}
                        break
            if re.search(r'(s\.r\.o\.|a\.s\.|llc|inc|gmbh|ltd)', lt_lower):
                fw, lw = line['words'][0], line['words'][-1]
                if "nazev" not in data["odberatel"]:
                    data["odberatel"]["nazev"] = {"hodnota": lt, "box": [fw['x'], fw['y'], (lw['x']+lw['w'])-fw['x'], max(fw['h'], lw['h'])]}
            in_customer_block -= 1
            continue

        # Číslo faktury
        m = re.search(r'(?:faktura\s*[č\.]|invoice\s*(?:no\.?|#)|cislo\s+faktury)\s*:?\s*([A-Z0-9\-/]+)', lt, re.IGNORECASE)
        if m and "cislo_faktury" not in data["faktura"]:
            fw, lw = line['words'][0], line['words'][-1]
            data["faktura"]["cislo_faktury"] = {"hodnota": m.group(1), "box": [fw['x'], fw['y'], (lw['x']+lw['w'])-fw['x'], max(fw['h'], lw['h'])]}

        # Datumy
        for pat, key in [
            (r'(?:datum\s+splatnosti|due\s+(?:on|date))\s*:?\s*(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}|[A-Z][a-z]{2,8}\s+\d{1,2},?\s+\d{4})', "datum_splatnosti"),
            (r'(?:datum\s+(?:vystaveni|vystaven\u00ed)|issued\s+on)\s*:?\s*(\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}|[A-Z][a-z]{2,8}\s+\d{1,2},?\s+\d{4})', "datum_vystaveni"),
        ]:
            dm = re.search(pat, lt, re.IGNORECASE)
            if dm and key not in data["faktura"]:
                fw, lw = line['words'][0], line['words'][-1]
                data["faktura"][key] = {"hodnota": dm.group(1), "box": [fw['x'], fw['y'], (lw['x']+lw['w'])-fw['x'], max(fw['h'], lw['h'])]}

        # IČO dodavatele
        if re.search(r'\b(ic|ico|reg\.?\s*no\.?)\b', lt_lower):
            for w in line['words']:
                if re.match(r'^\d{8}$', w['text']) and "ico" not in data["prodavajici"]:
                    data["prodavajici"]["ico"] = {"hodnota": w['text'], "box": [w['x'], w['y'], w['w'], w['h']]}
                    ares = validate_ico_via_ares(w['text'])
                    if ares:
                        data["prodavajici"]["nazev_ares"] = {"hodnota": ares["nazev"], "box": []}
                        data["prodavajici"]["adresa_ares"] = {"hodnota": ares["adresa"], "box": []}
                        data["prodavajici"]["dic_ares"] = {"hodnota": ares.get("dic", ""), "box": []}
                    break

        # PSČ (přeskočíme řádky s cenami)
        line_has_price = bool(re.search(r'[\$€£]?\d+[.,]\d{2}', lt) and len(line['words']) > 3)
        if "psc" not in data["prodavajici"] and not line_has_price:
            for j, w in enumerate(line['words']):
                if re.match(r'^\d{5}$', w['text']):
                    data["prodavajici"]["psc"] = {"hodnota": w['text'], "box": [w['x'], w['y'], w['w'], w['h']]}
                    break
                elif re.match(r'^\d{3}$', w['text']) and j + 1 < len(line['words']) and re.match(r'^\d{2}$', line['words'][j + 1]['text']):
                    w2 = line['words'][j + 1]
                    data["prodavajici"]["psc"] = {"hodnota": f"{w['text']} {w2['text']}", "box": [w['x'], w['y'], (w2['x']+w2['w'])-w['x'], max(w['h'], w2['h'])]}
                    break

        # Název dodavatele
        if "nazev" not in data["prodavajici"] and i < 12:
            if re.search(r'(s\.r\.o\.|a\.s\.|llc|inc|gmbh|ltd)', lt_lower):
                fw, lw = line['words'][0], line['words'][-1]
                nazev_raw = lt
                for sep in ['|', ' OFFICIAL', ' ELECTRONIC']:
                    nazev_raw = nazev_raw.split(sep)[0]
                nazev_raw = re.sub(r'^(?:dodavatel|supplier)\s*:?\s*', '', nazev_raw, flags=re.IGNORECASE).strip()
                data["prodavajici"]["nazev"] = {"hodnota": nazev_raw, "box": [fw['x'], fw['y'], (lw['x']+lw['w'])-fw['x'], max(fw['h'], lw['h'])]}

        # Celková cena
        is_subtotal = bool(re.search(r'\bsub.?total\b', lt_lower_na))
        is_total = bool(re.search(r'\b(amount\s+to\s+pay|celkem\s+k\s+uhrade|fakturace\s+celkem|total:?|celkem:?)\b', lt_lower_na))
        if is_total and not is_subtotal or (is_subtotal and "celkova_cena" not in data["celkem"]):
            best_val, best_box = None, None
            for j in range(i, min(i + 2, len(lines))):
                for w_idx, w in enumerate(lines[j]['words']):
                    cand = w['text']
                    if not re.match(r'^[\$€£]?\d', cand):
                        continue
                    lookahead = cand
                    box_right = w['x'] + w['w']
                    for k in range(w_idx + 1, min(len(lines[j]['words']), w_idx + 3)):
                        nw = lines[j]['words'][k]
                        if re.match(r'^[\d\.,]+$', nw['text']) or nw['text'].upper() in ['CZK', 'KČ', 'EUR', 'USD', '$', '€', '£']:
                            lookahead += " " + nw['text']
                            box_right = nw['x'] + nw['w']
                        else:
                            break
                    m = re.search(r'([\d\s]+[.,]\d{2}(?:\s*[a-zA-Z$€£]+)?|(?:[$€£])[\d\s,]+[.,]\d{2})', lookahead)
                    if m:
                        val = m.group(1).strip()
                        has_currency = bool(re.search(r'[a-zA-Z$€£]', lookahead))
                        if best_val is None or has_currency:
                            best_val = val
                            best_box = [w['x'], w['y'], box_right - w['x'], w['h']]
            if best_val and (is_total or "celkova_cena" not in data["celkem"]):
                data["celkem"]["celkova_cena"] = {"hodnota": best_val, "box": best_box}

    # Produkty
    row_index = 1
    for line in lines:
        lt = line['text_united']
        lt_lower = lt.lower()
        lt_lower_na = remove_accents(lt_lower)
        is_vat_summary = bool(re.match(r'^\d{1,2}\s*%', lt.strip()) and not re.search(r'\b(ks|pcs|kusů|qty)\b', lt_lower))
        is_header = bool(re.search(r'\b(vat\s+rate|vat\s+base|unit\s+price|price\s+\(ex|products\s+model|oznaceni|kod)\b', lt_lower_na))
        is_product_row = (not is_vat_summary and not is_header and (
            re.search(r'\b(ks|pcs|kusů|qty)\b', lt_lower) or
            re.match(r'^\d+\s*x\s', lt_lower) or
            (re.search(r'\d+\s*%', lt) and re.search(r'[\$€£]?[\d,]+[.,]\d{2}', lt) and len(line['words']) > 4)
        ))
        if is_product_row:
            fw, lw = line['words'][0], line['words'][-1]
            price_pattern = r'[\$€£]\s*[\d]{1,3}(?:[\s,][\d]{3})*[.,]\d{2}|[\d]{1,3}(?:[,][\d]{3})*[.][\d]{2}(?:\s*(?:CZK|Kč|EUR|USD|GBP))?|[\d]{1,3}(?:[\s][\d]{3})+[.,]\d{2}(?:\s*(?:CZK|Kč|EUR|USD|GBP))?|[\d]+[.,]\d{2}\s*(?:CZK|Kč|EUR|USD|GBP)'
            prices = [p.strip() for p in re.findall(price_pattern, lt) if p.strip()]
            item = {
                "indicie_radku": row_index,
                "radek_raw": {"hodnota": lt, "box": [fw['x'], fw['y'], (lw['x']+lw['w'])-fw['x'], max(w['h'] for w in line['words'])]},
            }
            qty_match = re.match(r'^(\d+)[\.,]?\d*\s*(?:ks|pcs|x|kusů)?\s+', lt, re.IGNORECASE)
            if qty_match:
                item["pocet"] = {"hodnota": qty_match.group(1), "box": []}
            dph_match = re.search(r'(\d{1,2})\s*%', lt)
            if dph_match:
                item["dph_sazba"] = {"hodnota": dph_match.group(1) + "%", "box": []}
            if len(prices) >= 2:
                item["cena_za_kus"] = {"hodnota": prices[-2], "box": []}
                item["cena_celkem"] = {"hodnota": prices[-1], "box": []}
            elif len(prices) == 1:
                item["cena_celkem"] = {"hodnota": prices[0], "box": []}
            name_match = re.match(r'^\d+[\.,]?\d*\s*(?:ks|pcs|x|kusů)?\s+(.+?)(?:\s+\d{1,2}\s*%|\s+[\$€£]?\d)', lt, re.IGNORECASE)
            if name_match:
                item["nazev"] = {"hodnota": name_match.group(1).strip(), "box": []}
            data["produkty"].append(item)
            row_index += 1

    return data


def extract_data_with_coords(img_array):
    """
    Hlavní vstupní bod.
    1. Tesseract OCR → text + souřadnice
    2. Zkus najít IČO
    3. Pokud existuje šablona pro toto IČO → aplikuj šablonu (deterministicky!)
    4. Jinak → heuristická extrakce + auto-generace šablony kandidáta
    """
    d = pytesseract.image_to_data(img_array, lang='ces', output_type=Output.DICT)
    all_text = " ".join([str(w) for w in d['text'] if str(w).strip()])
    lines = build_lines_from_ocr(d)
    img_shape = img_array.shape

    # 2. Zkus IČO
    ico = find_ico_in_lines(lines)

    # 3. Template mode (pokud existuje)
    if ico and store.load(ico):
        template = store.load(ico)
        template_result = apply_template(template, lines, img_shape)
        structured = {
            "metadata": {"typ_dokumentu": classify_doc(all_text), "stranky": 1, "rezim": "TEMPLATE", "ico_key": ico},
            "sablona_ico": ico,
            "sablona_verze": template.get("meta", {}).get("version", 1),
        }
        structured.update(template_result)
        return structured, d, {"mode": "template", "ico": ico}

    # 4. Heuristická extrakce + kandidátská šablona (pro Human review)
    data = heuristic_extract(lines, img_shape, all_text)
    data["metadata"]["rezim"] = "HEURISTIKA"

    # Připrav kandidáta šablony (uloží se až po human-review potvrzení)
    kv_pairs = detect_kv_pairs(lines)
    table_box = detect_table_region(img_array)
    issuer = data.get("prodavajici", {}).get("nazev", {}).get("hodnota", "")
    candidate_template = generate_template(kv_pairs, table_box, img_shape, ico or "unknown", issuer)

    meta = {
        "mode": "heuristic",
        "ico": ico,
        "candidate_template": candidate_template,
        "kv_pairs": kv_pairs,
        "table_box": table_box,
    }

    return data, d, meta
