"""
template_engine.py – auto-generace YAML šablon z Tesseract výstupu

Architektura:
1. detect_kv_pairs(lines)      – najde label:hodnota páry ze souřadnic
2. detect_table_region(img)    – OpenCV hledá tabulkovou oblast
3. generate_template(...)      – vygeneruje YAML šablonu
4. apply_template(template, lines, img_shape) – aplikuje šablonu na nový dokument
5. TemplateStore               – uložení/načtení šablon podle IČO
"""

import os
import re
import yaml
import cv2
import numpy as np
import unicodedata

TEMPLATES_DIR = os.path.join(os.path.dirname(__file__), "templates")
os.makedirs(TEMPLATES_DIR, exist_ok=True)


def remove_accents(s):
    """Odstraní českou diakritiku pro robustní porovnávání textu."""
    if not isinstance(s, str): return str(s)
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ---------------------------------------------------------------------------
# 1. KV-pair detektor
# ---------------------------------------------------------------------------

def detect_kv_pairs(lines):
    """
    Analyzuje řádky textu a hledá dvojice Klíč:Hodnota (KV-pairs).
    Slouží jako základ pro automatické generování šablon.
    
    Args:
        lines (list): Seskupené řádky z build_lines_from_ocr.
        
    Returns:
        list: Seznam nalezených párů {label, value, label_box, value_box, line_idx}.
    """
    kv_pairs = []

    for i, line in enumerate(lines):
        text = line['text_united']
        words = line['words']

        # Procházíme slova a hledáme ta, která končí ':' (kotvy)
        for j, w in enumerate(words):
            wt = w['text'].rstrip(':').strip()
            if not wt or len(wt) < 2:
                continue

            # Kontrola, zda slovo vypadá jako label (buď končí ':' nebo patří mezi známá klíčová slova)
            is_anchor = w['text'].endswith(':') or remove_accents(wt.lower()) in [
                'faktura', 'invoice', 'ic', 'ico', 'dic', 'vat id', 'reg no', 'reference',
                'total', 'celkem', 'datum', 'due', 'issued', 'splatnosti', 'vystaveni',
                'dodavatel', 'supplier', 'odberatel', 'customer', 'amount'
            ]
            if not is_anchor:
                continue

            # Jako hodnotu (value) bereme slova bezprostředně vpravo na stejném řádku
            value_words = words[j + 1:]
            if not value_words:
                # Pokud vpravo nic není, zkusíme první slovo na dalším řádku
                if i + 1 < len(lines) and lines[i + 1]['words']:
                    value_words = [lines[i + 1]['words'][0]]

            if value_words:
                vw = value_words[0]
                # Spojíme až 4 slova za kotvou jako výslednou hodnotu
                value_text = " ".join(v['text'] for v in value_words[:4])
                kv_pairs.append({
                    "label": wt,
                    "value": value_text.strip(),
                    "label_box": [w['x'], w['y'], w['w'], w['h']],
                    "value_box": [vw['x'], vw['y'], vw['w'], vw['h']],
                    "line_idx": i,
                })

    return kv_pairs


# ---------------------------------------------------------------------------
# 2. Detekce tabulkové oblasti (OpenCV)
# ---------------------------------------------------------------------------

def detect_table_region(img_array):
    """
    Pokročilá detekce tabulky v dokumentu pomocí počítačového vidění (OpenCV).
    Hledá oblasti s hustým výskytem vodorovných a svislých čar (např. mřížky).
    
    Args:
        img_array (np.ndarray): Snímek faktury v RGB.
        
    Returns:
        list|None: Souřadnice tabulky [x, y, w, h] v pixelech.
    """
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # Prahování: tmavé čáry na světlém pozadí se převedou na bílou pro morfologické operace
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    img_h, img_w = gray.shape

    # Detekce vodorovných čar (minimální délka 33 % šířky dokumentu)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (img_w // 3, 1))
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel)

    # Detekce svislých čar (minimální výška 3.3 % výšky dokumentu)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, img_h // 30))
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kernel)

    # Sloučení vodorovných a svislých čar do jedné masky (mřížky)
    table_mask = cv2.add(horizontal, vertical)
    # Nalezení vnějších obrysů výsledné mřížky
    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Vybereme největší souvislou oblast (pravděpodobně hlavní tabulka produktů)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    
    # Validace: tabulka musí zabírat alespoň 5 % plochy dokumentu
    if w * h < (img_w * img_h * 0.05):
        return None

    return [x, y, w, h]


def detect_table_rows_in_region(lines, table_box, img_shape):
    """
    Filtruje řádky textu, které se geometricky nacházejí uvnitř detekované tabulkové oblasti.
    
    Args:
        lines (list): Všechny řádky z OCR.
        table_box (list): [x, y, w, h] tabulky.
        img_shape (tuple): Rozměry obrázku.
        
    Returns:
        list: Řádky uvnitř tabulky.
    """
    if not table_box:
        return []
    tx, ty, tw, th = table_box
    img_h, img_w = img_shape[:2]

    table_lines = []
    for line in lines:
        if not line['words']:
            continue
        # Kontrola, zda první slovo řádku spadá do vertikálního rozsahu tabulky (pro jednoduchost)
        line_y = line['words'][0]['y']
        if ty <= line_y <= ty + th:
            table_lines.append(line)
    return table_lines


# ---------------------------------------------------------------------------
# 3. Generátor YAML šablon
# ---------------------------------------------------------------------------

def _rel(val, dim):
    """Převede pixelovou souřadnici na relativní (0-1)."""
    return round(val / dim, 4) if dim else 0.0


def generate_template(kv_pairs, table_box, img_shape, ico, issuer_name=""):
    """
    Vytvoří definici YAML šablony na základě nalezených kotev a geometrie dokumentu.
    Tato šablona umožňuje 100% přesnou extrakci pro identické typy dokumentů.
    
    Args:
        kv_pairs (list): Nalezené dvojice štítek:hodnota.
        table_box (list): Detekovaná oblast tabulky.
        img_shape (tuple): Rozměry dokumentu.
        ico (str): Unikátní identifikátor (IČO) dodavatele.
        issuer_name (str): Název firmy.
        
    Returns:
        dict: Strukturovaná šablona pro TemplateStore.
    """
    img_h, img_w = img_shape[:2]

    template = {
        "meta": {
            "ico": ico,
            "issuer": issuer_name,
            "version": 1,
        },
        "fields": {},
        "table": None,
    }

    # Mapování label → klíč JSONu
    label_to_key = {
        "faktura": "cislo_faktury",
        "invoice": "cislo_faktury",
        "ic": "ico",
        "ico": "ico",
        "dic": "dic",
        "vat id": "dic",
        "reg no": "ico",
        "total": "celkova_cena",
        "celkem": "celkova_cena",
        "amount": "celkova_cena",
        "datum": "datum_vystaveni",
        "issued": "datum_vystaveni",
        "due": "datum_splatnosti",
        "splatnosti": "datum_splatnosti",
        "reference": "variabilni_symbol",
        "dodavatel": "prodavajici_nazev",
        "supplier": "prodavajici_nazev",
        "odberatel": "odberatel_nazev",
        "customer": "odberatel_nazev",
    }

    for kv in kv_pairs:
        label_na = remove_accents(kv["label"].lower())
        json_key = None

        for pattern, key in label_to_key.items():
            if pattern in label_na:
                json_key = key
                break

        if not json_key:
            # Generický klíč ze znaku label
            json_key = re.sub(r"[^a-z0-9]", "_", label_na)

        if json_key in template["fields"]:
            continue  # Duplikát – přeskočíme

        lb = kv["label_box"]
        vb = kv["value_box"]

        template["fields"][json_key] = {
            # Relativní pozice value boxu
            "zone": {
                "x": _rel(vb[0], img_w),
                "y": _rel(vb[1], img_h),
                "w": _rel(vb[2], img_w),
                "h": _rel(vb[3], img_h),
            },
            # Relativní pozice anchor (label)
            "anchor": {
                "label": kv["label"],
                "x": _rel(lb[0], img_w),
                "y": _rel(lb[1], img_h),
            },
            "value_snapshot": kv["value"],  # snapshot aktuální hodnoty pro ladění
        }

    # Tabulka
    if table_box:
        tx, ty, tw, th = table_box
        template["table"] = {
            "zone": {
                "x": _rel(tx, img_w),
                "y": _rel(ty, img_h),
                "w": _rel(tw, img_w),
                "h": _rel(th, img_h),
            }
        }

    return template


# ---------------------------------------------------------------------------
# 4. Aplikátor šablon
# ---------------------------------------------------------------------------

def apply_template(template, lines, img_shape):
    """
    Aplikuje uloženou šablonu na OCR data nového dokumentu stejného typu.
    Slouží pro ultra-rychlou a přesnou extrakci bez potřeby AI.
    
    Args:
        template (dict): Načtená YAML šablona.
        lines (list): OCR řádky nového dokumentu.
        img_shape (tuple): Rozměry nového dokumentu.
        
    Returns:
        dict: Extrahovaná pole s jejich souřadnicemi.
    """
    img_h, img_w = img_shape[:2]
    result = {}

    for field_key, field_def in template.get("fields", {}).items():
        zone = field_def.get("zone", {})
        # Absolutní souřadnice cílové zóny přepočtené z relativních (%)
        zx = zone.get("x", 0) * img_w
        zy = zone.get("y", 0) * img_h
        zw = zone.get("w", 0.3) * img_w
        zh = zone.get("h", 0.05) * img_h

        # Tolerance pro případ mírného posunu dokumentu při skenování
        y_tol = max(zh * 1.5, img_h * 0.02)
        x_tol = max(zw * 1.5, img_w * 0.05)

        # Hledáme všechna slova, která geometricky spadají do definované zóny
        matched_words = []
        for line in lines:
            for w in line['words']:
                wx, wy = w['x'], w['y']
                if (zx - x_tol) <= wx <= (zx + zw + x_tol) and (zy - y_tol) <= wy <= (zy + zh + y_tol):
                    matched_words.append(w)

        if matched_words:
            # Seřazení slov zleva doprava a spojení do textu
            matched_words.sort(key=lambda w: w['x'])
            value_text = " ".join(w['text'] for w in matched_words)
            first_w = matched_words[0]
            last_w = matched_words[-1]
            result[field_key] = {
                "hodnota": value_text.strip(),
                "box": [first_w['x'], first_w['y'],
                        (last_w['x'] + last_w['w']) - first_w['x'],
                        max(w['h'] for w in matched_words)],
                "source": "template",
            }

    return result


# ---------------------------------------------------------------------------
# 5. TemplateStore – uložení a načtení šablon
# ---------------------------------------------------------------------------

class TemplateStore:
    """
    Správce úložiště YAML šablon.
    Umožňuje perzistentní ukládání a načítání šablon pro konkrétní firmy (IČO).
    """
    def __init__(self, directory=TEMPLATES_DIR):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, ico):
        """Vytvoří bezpečnou cestu k souboru šablony."""
        safe = re.sub(r"[^a-zA-Z0-9]", "_", str(ico))
        return os.path.join(self.directory, f"{safe}.yaml")

    def save(self, ico, template):
        """Uloží definici šablony na disk jako lidskou čitelný YAML."""
        with open(self._path(ico), "w", encoding="utf-8") as f:
            yaml.dump(template, f, allow_unicode=True, sort_keys=False)

    def load(self, ico):
        """Vyhledá a načte šablonu pro dané IČO."""
        path = self._path(ico)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def list_templates(self):
        """Vrátí seznam všech dostupných IČO v databázi šablon."""
        icos = []
        for fname in os.listdir(self.directory):
            if fname.endswith(".yaml"):
                with open(os.path.join(self.directory, fname), "r") as f:
                    t = yaml.safe_load(f)
                    icos.append(t.get("meta", {}).get("ico", fname.replace(".yaml", "")))
        return icos

    def update_field(self, ico, field_key, corrected_value, new_box, img_shape):
        """
        Umožňuje doučování systému po korekci uživatelem.
        Aktualizuje zónu pole v šabloně tak, aby příště extrakce proběhla správně.
        """
        template = self.load(ico)
        if not template:
            return
        img_h, img_w = img_shape[:2]
        if field_key in template.get("fields", {}):
            x, y, w, h = new_box
            template["fields"][field_key]["zone"] = {
                "x": _rel(x, img_w), "y": _rel(y, img_h),
                "w": _rel(w, img_w), "h": _rel(h, img_h),
            }
            template["fields"][field_key]["value_snapshot"] = corrected_value
        self.save(ico, template)
