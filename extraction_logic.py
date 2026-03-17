import pytesseract
from pytesseract import Output
import re
import unicodedata
import requests
import google.generativeai as genai
import openai
import json
from thefuzz import fuzz, process

def fuzzy_find_labels(query, choices, threshold=85):
    """
    Vyhledá klíčová slova v seznamu candidats s využitím přibližné shody (Levenshteinova vzdálenost).
    
    Args:
        query (str): Hledaný výraz (např. 'Odběratel').
        choices (list): Seznam řetězců, ve kterých se má hledat (např. labels z OCR).
        threshold (int): Minimální skóre podobnosti (0-100), aby byl výsledek považován za shodu.
        
    Returns:
        list: Seznam odpovídajících řetězců, které překročily práh podobnosti.
    """
    results = process.extract(query, choices, scorer=fuzz.partial_ratio)
    return [res[0] for res in results if res[1] >= threshold]

def find_spatial_neighbors(anchor_word, all_words, direction='right', threshold=300):
    """
    Najde slova v geografické blízkosti kotevního slova (šablony) v určeném směru.
    Tato funkce je klíčová pro extrakci hodnot, které nejsou na stejném řádku nebo jsou odsazené.
    
    Args:
        anchor_word (dict): Slovník s informacemi o kotevním slově {'x', 'y', 'w', 'h', 'text'}.
        all_words (list): Seznam všech detekovaných slov v dokumentu.
        direction (str): Směr hledání - 'right' (vpravo) nebo 'down' (dolů).
        threshold (int): Maximální vzdálenost v pixelech.
        
    Returns:
        list: Seznam sousedních slov seřazených podle vzdálenosti od kotvy.
    """
    ax, ay, aw, ah = anchor_word['x'], anchor_word['y'], anchor_word['w'], anchor_word['h']
    acx, acy = ax + aw/2, ay + ah/2
    
    neighbors = []
    for w in all_words:
        # Přeskočíme samotnou kotvu, aby nenašla samu sebe
        if w['x'] == ax and w['y'] == ay: continue
        
        wcx, wcy = w['x'] + w['w']/2, w['y'] + w['h']/2
        
        # Výpočet euklidovské vzdálenosti středů
        dx = wcx - acx
        dy = wcy - acy
        dist = (dx**2 + dy**2)**0.5
        
        # Filtrujeme příliš vzdálená slova
        if dist > threshold: continue
        
        # Logika pro směr 'vpravo': musí být vpravo od kotvy a v podobné horizontální linii (tolerance ah)
        if direction == 'right' and dx > 0 and abs(dy) < ah:
            neighbors.append(w)
        # Logika pro směr 'dolů': musí být pod kotvou a v podobné vertikální linii (tolerance aw * 2)
        elif direction == 'down' and dy > 0 and abs(dx) < aw * 2:
            neighbors.append(w)
            
    # Seřadíme od nejbližšího, aby první výsledek byl nejrelevantnější
    neighbors.sort(key=lambda x: (x['x'] - ax)**2 + (x['y'] - ay)**2)
    return neighbors

def extract_with_llm(all_text, api_key):
    """
    Volá LLM (Google Gemini nebo OpenRouter/DeepSeek) k extrakci strukturovaných dat z textu.
    Funkce automaticky rozpozná typ klíče a zvolí odpovídajícího poskytovatele.
    
    Args:
        all_text (str): Kompletní text extrahovaný z OCR.
        api_key (str): API klíč (Gemini nebo OpenRouter 'sk-or-').
        
    Returns:
        dict: Strukturovaná data ve formátu JSON, nebo None při selhání.
    """
    if not api_key:
        return None

    # Definice promptu s jasnými pravidly pro normalizaci dat
    prompt = f"""
    Extract invoice data from the following OCR text into a clean, normalized JSON format.
    Return ONLY the raw JSON without any markdown formatting or explanations.
    
    Data normalization rules:
    - Dates: Format as YYYY-MM-DD.
    - Prices/Amounts: Provide numeric values (float/int) alongside the raw string.
    - Currency: Provide ISO currency code (CZK, USD, EUR, etc.).
    - Names/Addresses: Clean up any OCR artifacts.
    
    Expected JSON structure:
    {{
        "typ_dokumentu": "FAKTURA",
        "prodavajici": {{ 
            "nazev": "...", 
            "ico": "...", 
            "dic": "...",
            "adresa": {{ "ulice": "...", "mesto": "...", "psc": "...", "zeme": "..." }},
            "bankovni_spojeni": {{ "cislo_uctu": "...", "iban": "...", "swift": "..." }}
        }},
        "odberatel": {{ 
            "nazev": "...", 
            "ico": "...", 
            "dic": "...",
            "adresa": {{ "ulice": "...", "mesto": "...", "psc": "...", "zeme": "..." }}
        }},
        "faktura": {{ 
            "cislo_faktury": "...", 
            "variabilni_symbol": "...",
            "datum_vystaveni": "YYYY-MM-DD", 
            "datum_splatnosti": "YYYY-MM-DD",
            "zpusob_uhrady": "..."
        }},
        "produkty": [
            {{ 
                "nazev": "...", 
                "mnozstvi": 1.0, 
                "mj": "ks",
                "cena_za_kus": {{ "raw": "...", "hodnota": 0.0 }}, 
                "cena_celkem": {{ "raw": "...", "hodnota": 0.0 }},
                "dph_sazba": 21
            }}
        ],
        "celkem": {{ 
            "celkova_cena": {{ "raw": "...", "hodnota": 0.0, "mena": "..." }},
            "zaklad_dph": 0.0,
            "dph_celkem": 0.0
        }}
    }}
    
    OCR TEXT:
    {all_text}
    """

    # --- Varianta A: OpenRouter (OpenAI-compatible) ---
    if api_key.startswith("sk-or-"):
        try:
            client = openai.OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key
            )
            # Primární model: DeepSeek V3 (vysoký poměr kvalita/cena)
            response = client.chat.completions.create(
                model="deepseek/deepseek-chat", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            # Odstranění markdown obalů pro získání validního JSONu
            clean_json = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
            return json.loads(clean_json)
        except Exception as e:
            print(f"OpenRouter error: {e}")
            # Sekundární fallback na Gemini pro případ výpadku modelu na OpenRouteru
            try:
                response = client.chat.completions.create(
                    model="google/gemini-flash-1.5",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                clean_json = response.choices[0].message.content.replace('```json', '').replace('```', '').strip()
                return json.loads(clean_json)
            except Exception as e2:
                print(f"OpenRouter fallback error: {e2}")
                return None
    # --- Varianta B: Nativní Google Gemini ---
    else:
        genai.configure(api_key=api_key)
        # Zkoušíme různé generace modelu pro maximální kompatibilitu
        model_names = ['gemini-1.5-flash', 'gemini-flash-latest', 'gemini-2.0-flash']
        last_err = None
        for name in model_names:
            try:
                model = genai.GenerativeModel(name)
                response = model.generate_content(prompt)
                clean_json = response.text.replace('```json', '').replace('```', '').strip()
                return json.loads(clean_json)
            except Exception as e:
                last_err = e
                continue
        print(f"Gemini error (all models failed): {last_err}")
        return None

def find_text_box(target_text, ocr_data):
    """
    Pokusí se najít souřadnice (bounding box) pro konkrétní textový řetězec v OCR datech.
    Podporuje více-slovné názvy tím, že hledá sekvenci slov, která se nejlépe shoduje.
    
    Args:
        target_text (str): Text, pro který hledáme souřadnice (např. 'Prodavač').
        ocr_data (dict): Data z pytesseract.image_to_data.
        
    Returns:
        list: [left, top, width, height] nebo prázdný list.
    """
    if not target_text or len(str(target_text)) < 2:
        return []
    
    # Normalizace vstupu na malá písmena a rozdělení na slova
    target_str = str(target_text).strip()
    target_words = target_str.lower().split()
    if not target_words: return []

    # Seznam všech slov z OCR
    ocr_words = [str(t).lower() for t in ocr_data['text']]
    
    # --- Metoda 1: Sekvenční shoda (přesná sekvence slov) ---
    for i in range(len(ocr_words) - len(target_words) + 1):
        match = True
        for j in range(len(target_words)):
            # Kontrolujeme, zda je hledané slovo obsaženo v OCR slově (proti artefaktům)
            if target_words[j] not in ocr_words[i+j]:
                match = False
                break
        
        if match:
            # Výpočet společného rámečku pro všechna slova v sekvenci
            left = min(ocr_data['left'][i:i+len(target_words)])
            top = min(ocr_data['top'][i:i+len(target_words)])
            right = max(ocr_data['left'][k] + ocr_data['width'][k] for k in range(i, i+len(target_words)))
            bottom = max(ocr_data['top'][k] + ocr_data['height'][k] for k in range(i, i+len(target_words)))
            return [left, top, right - left, bottom - top]

    # --- Metoda 2: Fallback na přibližnou shodu v celých slovech ---
    target_na = remove_accents(target_str.lower())
    for i, t in enumerate(ocr_data['text']):
        t_clean = remove_accents(str(t).lower())
        # Pokud se texty výrazně překrývají a slovo je aspoň trochu dlouhé
        if len(t_clean) > 3 and (t_clean in target_na or target_na in t_clean):
            return [ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i]]
            
    return []

def map_json_to_boxes(json_data, ocr_data):
    """
    Rekurzivně prochází strukturu JSONu (z LLM) a doplňuje souřadnice (boxy) z OCR dat.
    Tato funkce transformuje prosté hodnoty na objekty obsahující i geometrickou polohu.
    
    Args:
        json_data (dict|list): Výstup z LLM (např. extrahované produkty, adresy).
        ocr_data (dict): Surová data z Tesseractu (left, top, width, height, text).
        
    Returns:
        dict|list: Modifikovaná data, kde textové hodnoty jsou doplněny o 'box'.
    """
    if isinstance(json_data, dict):
        # Pokud je to již objekt s kotevním boxem, neprovádíme další mapování
        if "box" in json_data and ("hodnota" in json_data or "raw" in json_data):
            return json_data

        new_dict = {}
        # Prioritní pole, která chceme vizualizovat (mají vliv na back-mapping)
        for k, v in json_data.items():
            if k in ["raw", "nazev", "cislo_faktury", "ico", "dic", "mesto", "ulice"] and isinstance(v, str):
                new_dict[k] = {
                    "hodnota": v,
                    "box": find_text_box(v, ocr_data)
                }
            elif isinstance(v, (dict, list)):
                # Rekurzivní zpracování vnořených struktur
                new_dict[k] = map_json_to_boxes(v, ocr_data)
            else:
                # Ostatní textové hodnoty (pokud to nejsou systémové klíče) mapujeme také
                if k not in ["box", "metadata", "mena", "mnozstvi", "dph_sazba"]:
                     new_dict[k] = {
                        "hodnota": str(v),
                        "box": find_text_box(v, ocr_data)
                     }
                else:
                    new_dict[k] = v
        return new_dict
    elif isinstance(json_data, list):
        # Hromadné zpracování seznamů (např. řádky faktury)
        return [map_json_to_boxes(item, ocr_data) for item in json_data]
    return json_data

from template_engine import (
    TemplateStore,
    detect_kv_pairs,
    detect_table_region,
    detect_table_rows_in_region,
    generate_template,
    apply_template,
)

import os
import sys

# Tesseract command configuration
if sys.platform.startswith('win'):
    # Common Windows installation paths
    tesseract_win_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
    ]
    for p in tesseract_win_paths:
        if os.path.exists(p):
            pytesseract.pytesseract.tesseract_cmd = p
            break
else:
    # Default/Homebrew path for macOS/Linux
    if os.path.exists('/opt/homebrew/bin/tesseract'):
        pytesseract.pytesseract.tesseract_cmd = '/opt/homebrew/bin/tesseract'

store = TemplateStore()


def remove_accents(input_str):
    """Odstraní českou diakritiku (např. 'ěščř' -> 'escr')."""
    nfkd_form = unicodedata.normalize('NFKD', str(input_str))
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])


def classify_doc(text):
    """
    Sémanticky klasifikuje typ dokumentu na základě přítomnosti klíčových slov.
    
    Returns:
        str: "FAKTURA", "NABÍDKA" nebo "EMAIL/OSTATNÍ".
    """
    text_lower = remove_accents(text.lower())
    if "faktura" in text_lower or "danovy doklad" in text_lower or "invoice" in text_lower:
        return "FAKTURA"
    elif "nabidka" in text_lower or "cenova kalkulace" in text_lower or "quote" in text_lower or "estimate" in text_lower:
        return "NABÍDKA"
    else:
        return "EMAIL/OSTATNÍ"


def validate_ico_via_ares(ico):
    """
    Ověří pravdivost IČO a stáhne oficiální data o firmě z vládního registru ARES.
    
    Args:
        ico (str): Osmimístné IČO.
        
    Returns:
        dict|None: Data firmy (název, adresa, dič) nebo None při selhání.
    """
    try:
        # Volání veřejného REST API systému ARES (nová verze 2024)
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
        # V případě výpadku ARES nebo nevalidního IČO vracíme None
        pass
    return None


def build_lines_from_ocr(d):
    """
    Seskupí jednotlivá slova z Tesseractu do logických řádků podle jejich Y-ové souřadnice.
    Tato funkce je nezbytná pro správné fungování prostorových a regex pravidiel.
    
    Args:
        d (dict): Výstup z pytesseract.image_to_data.
        
    Returns:
        list: Seznam řádků se sjednoceným textem a hromadnými souřadnicemi.
    """
    words = []
    # Nejprve extrahujeme všechna neprázdná slova a vypočítáme jejich vizuální střed
    for i in range(len(d['text'])):
        text = str(d['text'][i]).strip()
        if not text:
            continue
        left, top, width, height = d['left'][i], d['top'][i], d['width'][i], d['height'][i]
        center_y = top + height / 2.0
        words.append({"text": text, "x": left, "y": top, "w": width, "h": height, "cy": center_y})

    # Seřadíme slova podle svislé osy
    words.sort(key=lambda w: w['cy'])
    
    lines = []
    for w in words:
        # Pokud je slovo na podobné Y úrovni jako předchozí řádek, přidáme ho do něj
        if lines and abs(lines[-1]['cy'] - w['cy']) < 15:
            lines[-1]['words'].append(w)
            # Průběžně aktualizujeme průměrnou Y souřadnici řádku
            n = len(lines[-1]['words'])
            lines[-1]['cy'] = (lines[-1]['cy'] * (n - 1) + w['cy']) / n
        else:
            # Jinak vytvoříme řádek nový
            lines.append({'cy': w['cy'], 'words': [w]})

    # Pro každý řádek seřadíme slova zleva doprava a spojíme je do textu
    for line in lines:
        line['words'].sort(key=lambda w: w['x'])
        line['text_united'] = " ".join([w['text'] for w in line['words']])

    return lines


def find_ico_in_lines(lines):
    """
    Rychlá detekce IČO dodavatele přímo v textu pomocí regulárních výrazů.
    Hledá osmimístné číslo v blízkosti klíčových slov jako 'IČ' nebo 'Reg.no'.
    """
    for line in lines:
        lt = line['text_united'].lower()
        if re.search(r'\b(ic|ico|reg\.?\s*no\.?)\b', lt):
            for w in line['words']:
                # IČO musí být přesně 8 číslic
                if re.match(r'^\d{8}$', w['text']):
                    return w['text']
    return None


def heuristic_extract(lines, img_shape, all_text):
    """
    Implementace komplexní heuristické extrakce (bez využití AI).
    Využívá kombinaci regulárních výrazů, fuzzy matchingu a geometrického vyhledávání.
    Tento algoritmus slouží jako robustní fallback nebo základ pro generování šablon.
    
    Args:
        lines (list): Seskupené řádky textu.
        img_shape (tuple): Rozměry obrázku (v, š).
        all_text (str): Kompletní text dokumentu.
        
    Returns:
        dict: Extrahovaná data v enterprise JSON formátu.
    """
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
    in_supplier_block = 0

    # Všechna slova pro prostorové hledání
    all_words = []
    for line in lines:
        all_words.extend(line['words'])

    for i, line in enumerate(lines):
        lt = line['text_united']
        lt_lower = lt.lower()
        lt_lower_na = remove_accents(lt_lower)

        # --- 1. ODBĚRATEL (Fuzzy + Spatial) ---
        customer_labels = ["odberatel", "customer", "bill to", "sold to", "prijemce", "ship to", "recipient"]
        # Detekce řádku s označením odběratele pomocí fuzzy matching
        is_customer_row = any(fuzz.partial_ratio(lab, lt_lower_na) > 85 for lab in customer_labels)
        if is_customer_row:
            in_customer_block = 6 # Předpokládáme, že adresa následuje v dalších 6 řádcích
            # SPECIAL: Pokud je na řádku i "Místo určení", vezmeme jen část po "Odběratel"
            work_text = lt
            # Hledáme "odberatel" i se záměnou diakritiky
            for sep in ["odběratel", "odberatel"]:
                if remove_accents(sep) in lt_lower_na:
                     m_split = re.split(fr'{sep}\s*:?', lt, flags=re.IGNORECASE)
                     if len(m_split) > 1:
                         # Pokud je tam i "Místo určení", vezmeme tu část za Odběratelem
                         work_text = m_split[-1].strip()
                         break
            
            clean_name = re.sub(r'^(?:odberatel|customer|bill to|sold to|prijemce|ship to|recipient)\s*:?\s*', '', work_text, flags=re.IGNORECASE).strip()
            # Odstraníme "Místo určení:" prefix pokud zbyl na začátku
            clean_name = re.sub(r'^misto\s+urceni\s*:?\s*', '', clean_name, flags=re.IGNORECASE)
            if len(clean_name) > 3:
                # Pokud je název na stejném řádku, vezmeme jeho souřadnice
                fw, lw = line['words'][0], line['words'][-1]
                data["odberatel"]["nazev"] = {"hodnota": clean_name, "box": [fw['x'], fw['y'], (lw['x']+lw['w'])-fw['x'], max(w['h'] for w in line['words'])]}
            else:
                # Pokud je název pod štítkem, použijeme prostorové vyhledávání (Spatial Neighbors)
                anchor = line['words'][-1]
                neighbors = find_spatial_neighbors(anchor, all_words, direction='down')
                if neighbors:
                    nb = neighbors[0]
                    data["odberatel"]["nazev"] = {"hodnota": nb['text'], "box": [nb['x'], nb['y'], nb['w'], nb['h']]}

        if in_customer_block > 0:
            # IČO
            ico_m = re.search(r'\b\d{8}\b', lt)
            if ico_m and any(fuzz.partial_ratio(lab, lt_lower) > 80 for lab in ["ico", "ic", "reg no"]):
                 data["odberatel"]["ico"] = {"hodnota": ico_m.group(0), "box": [line['words'][0]['x'], line['words'][0]['y'], 80, 20]}
            in_customer_block -= 1

        # --- 2. DODAVATEL ---
        supplier_labels = ["dodavatel", "supplier", "bill from", "issuer", "pay to", "sender"]
        is_supplier_row = any(fuzz.partial_ratio(lab, lt_lower_na) > 85 for lab in supplier_labels)
        if is_supplier_row:
            in_supplier_block = 6
            # Extrakce názvu dodavatele z řádku
            clean_name = re.sub(r'^(?:dodavatel|supplier|bill from|issuer|vystavil|pay to|sender)\s*:?\s*', '', lt, flags=re.IGNORECASE).strip()
            if len(clean_name) > 3:
                fw, lw = line['words'][0], line['words'][-1]
                data["prodavajici"]["nazev"] = {
                    "hodnota": clean_name.split('|')[0].strip(), 
                    "box": [fw['x'], fw['y'], (lw['x']+lw['w'])-fw['x'], max(w['h'] for w in line['words'])]
                }

        # --- 3. ČÍSLO FAKTURY ---
        invoice_keywords = ["faktura c", "invoice no", "invoice #", "cislo dokladu"]
        symbol_keywords = ["variabilni symbol", "symbol", "vs :"]
        is_invoice_label = any(fuzz.partial_ratio(lab, lt_lower_na) > 85 for lab in invoice_keywords)
        is_symbol_label = any(fuzz.partial_ratio(lab, lt_lower_na) > 85 for lab in symbol_keywords)

        if is_invoice_label or (is_symbol_label and "cislo_faktury" not in data["faktura"]):
            # Preventivní odstranění běžných artefaktů v číslování
            clean_lt = re.sub(r'\b(no\.?|nr\.?|#)\b', '', lt, flags=re.IGNORECASE)
            # Regex hledá kód složený z číslic, písmen a dělítek (např. 2024-001)
            m = re.search(r'(?:faktura|invoice|cislo|doklad|symbol)\s*[:#\.\s|]?\s*([A-Z0-9\-/]{3,20})', clean_lt, re.IGNORECASE)
            if m:
                val = m.group(1).strip()
                # Pokud regex zachytil jen slovo "FAKTURA", zkusíme najít hodnotu až za ním
                if val.lower() in ["faktura", "invoice", "doklad"]:
                     rem = clean_lt[m.end():].strip()
                     m2 = re.search(r'([A-Z0-9\-/]{5,})', rem)
                     if m2: val = m2.group(1)
                     else: val = None
                
                if val:
                    # Přednost má štítek "Faktura č." před "Variabilní symbol"
                    if is_symbol_label and data["faktura"].get("cislo_faktury", {}).get("method") == "primary": pass
                    else:
                        fw, lw = line['words'][0], line['words'][-1]
                        data["faktura"]["cislo_faktury"] = {
                            "hodnota": val, 
                            "box": [fw['x'], fw['y'], (lw['x']+lw['w'])-fw['x'], max(w['h'] for w in line['words'])],
                            "method": "primary" if is_invoice_label else "secondary"
                        }

        # --- 4. CELKOVÁ CENA ---
        if any(fuzz.partial_ratio(lab, lt_lower_na) > 85 for lab in ["celkem", "total", "amount due", "k uhrade", "pay", "fakturace"]):
            # Regex pro detekci částek s měnou (podporuje i symboly před/za číslem)
            price_regex = r'([\$€£Kč]?\s*[\d\s,]{2,}[.,]\d{2,4}(?:\s*[A-Z]{2,3}|\$|€|£|Kč)?)'
            prices_on_line = re.findall(price_regex, lt)
            if prices_on_line:
                # Pokud je na řádku víc cen (např. základ DPH a celkem), bereme tu největší
                best_p = max(prices_on_line, key=lambda p: float(re.sub(r'[^\d.]', '', p.replace(',', '').replace(' ', '')) or 0))
                fw, lw = line['words'][0], line['words'][-1]
                data["celkem"]["celkova_cena"] = {"hodnota": best_p, "box": [fw['x'], fw['y'], (lw['x']+lw['w'])-fw['x'], max(w['h'] for w in line['words'])]}

        # --- 5. IČO DODAVATEL (Patička / Spatial) ---
        ico_labels = ["ico", "ic", "reg no", "reg. no"]
        if any(fuzz.partial_ratio(lab, lt_lower) > 90 for lab in ico_labels) and in_customer_block <= 0:
             ico_m = re.search(r'\b\d{8}\b', lt)
             if ico_m and "ico" not in data["prodavajici"]:
                data["prodavajici"]["ico"] = {"hodnota": ico_m.group(0), "box": [line['words'][0]['x'], line['words'][0]['y'], 80, 20]}
                # Pokud jsme v dolní části dokumentu, zkusíme najít název firmy vedle IČO
                if i > len(lines) * 0.6:
                     neighbors = find_spatial_neighbors(line['words'][0], all_words, direction='right')
                     for nw in neighbors:
                         if len(nw['text']) > 4 and not re.search(r'\d', nw['text']):
                             data["prodavajici"]["nazev_footer"] = {"hodnota": nw['text'], "box": [nw['x'], nw['y'], nw['w'], nw['h']]}

    # --- 6. FALLBACK PRO DODAVATELE (Hlava/Pata) ---
    if "nazev" not in data["prodavajici"] or len(data["prodavajici"]["nazev"]["hodnota"]) < 4:
        if "nazev_footer" in data["prodavajici"]:
             # Pokud jsme našli dodavatele v patičce u IČO, použijeme ho
             data["prodavajici"]["nazev"] = data["prodavajici"]["nazev_footer"]
        elif "nazev" not in data["prodavajici"]:
            # Poslední záchrana: první řádek s textem, který nevypadá jako nadpis faktury
            for l in lines[:15]:
                text = l['text_united']
                is_junk = any(lab in text.lower() for lab in ["invoice", "faktura", "date", "datum", "due", "ordered", "tax", "customer", "odberatel", "prijemce", "doklad"])
                if len(l['words']) > 1 and not is_junk and not re.search(r'\d{4}', text):
                    fw, lw = l['words'][0], l['words'][-1]
                    data["prodavajici"]["nazev"] = {"hodnota": text.split('|')[0].strip(), "box": [fw['x'], fw['y'], (lw['x']+lw['w'])-fw['x'], max(w['h'] for w in l['words'])]}
                    break

    # --- 7. FALLBACK PRO CELKOVOU CENU (Hledání největší částky) ---
    if not data["celkem"].get("celkova_cena"):
        all_found_prices = []
        for l in lines[int(len(lines)*0.4):]:
            prm = re.findall(r'[\$€£Kč]?\s*[\d\s,]{3,}[.,]\d{2,4}(?:\s*CZK|\s*USD|\s*EUR)?', l['text_united'])
            for p in prm:
                clean_p = re.sub(r'[^\d.]', '', p.replace(',', '').replace(' ', ''))
                try: 
                    val_f = float(clean_p)
                    # Ignorujeme malé částky, hledáme hlavní total (nad 100)
                    if val_f > 100: all_found_prices.append((val_f, p, l))
                except: pass
        if all_found_prices:
            best = max(all_found_prices, key=lambda x: x[0])
            fw, lw = best[2]['words'][0], best[2]['words'][-1]
            data["celkem"]["celkova_cena"] = {"hodnota": best[1], "box": [fw['x'], fw['y'], (lw['x']+lw['w'])-fw['x'], max(w['h'] for w in best[2]['words'])]}

    # --- 8. PRODUKTY (Tabulková extrakce) ---
    row_index = 1
    for line in lines:
        lt = line['text_united']
        lt_lower = lt.lower()
        
        # Detekce řádku produktu: musí obsahovat množství/jednotku a aspoň jednu cenu
        # Regulární výraz pro množství: např. "5 ks", "10.5 m", "20 x"
        has_qty_unit = bool(re.search(r'\b(\d+(?:[.,\s]\d+)?)\s+(ks|pcs|kus|ks\.|mj|m|kg|bal|box|unit|qty)\b', lt_lower))
        has_qty_x = bool(re.search(r'\b(\d+(?:[.,\s]\d+)?)\s+x\s+', lt_lower))
        
        # Očištění cen od dlouhých interních ID (např. 1880639), která Tesseract chybně bere jako velké částky
        prices_on_line = []
        all_price_matches = re.findall(r'[\$€£Kč]?\s*[\d\s,]{1,8}[.,]\d{2,4}(?:\s*[A-Z]{3})?', lt)
        for p in all_price_matches:
            clean_p = re.sub(r'[^\d.]', '', p.replace(',', '').replace(' ', ''))
            # Pokud cena nemá oddělovač (tečku/čárku) a je velmi dlouhá, je to pravděpodobně ID produktu
            if len(clean_p) > 7 and '.' not in p and ',' not in p: continue
            prices_on_line.append(p)

        # Pokud řádek vypadá jako produkt
        if (has_qty_unit or has_qty_x or ("quanti" in lt_lower)) and len(prices_on_line) > 0:
            fw, lw = line['words'][0], line['words'][-1]
            item = {
                "indicie_radku": row_index,
                "radek_raw": {"hodnota": lt, "box": [fw['x'], fw['y'], (lw['x']+lw['w'])-fw['x'], max(w['h'] for w in line['words'])]},
            }
            
            # --- Extrakce názvu produktu ---
            # Název je typicky vše od začátku řádku až po první nalezenou cenu
            first_price_m = re.search(r'[\$€£Kč]?\s*[\d\s,]{1,8}[.,]\d{2,4}(?:\s*[A-Z]{3})?', lt)
            name_part = lt[:first_price_m.start()] if first_price_m else lt
            
            # Vyčištění názvu od množství a měrných jednotek na začátku
            name_clean = re.sub(r'^\s*\d+(?:[.,\s]\d+)?\s*(?:ks|pcs|kus|ks\.|mj|m|kg|bal|box|unit|qty|x)\b\s*', '', name_part, flags=re.IGNORECASE)
            # Odstranění dlouhých číselných kódů (ID)
            name_clean = re.sub(r'\b\d{6,}\b', '', name_clean)
            # Odstranění pořadových čísel (např. "1. ")
            name_clean = re.sub(r'^\s*\d+[\.\)]\s*', '', name_clean)
            # Odstranění DPH procent a technických slov
            name_clean = re.sub(r'\b\d{1,2}\s*%\b', '', name_clean) 
            name_clean = re.sub(r'\b(vat|unit|mj|cena|celkem|dph)\b', '', name_clean, flags=re.IGNORECASE)
            
            item["nazev"] = {"hodnota": name_clean.strip().lstrip('.:x- ©—| ').strip(), "box": []}
            
            # --- Extrakce cen (Jednotková vs Celková) ---
            if len(prices_on_line) >= 2:
                # Pokud máme dvě ceny, porovnáme je (jednotková bývá menší nebo má víc des. míst)
                p1_val = float(re.sub(r'[^\d.]', '', prices_on_line[-2].replace(',', '').replace(' ', '')) or 0)
                p2_val = float(re.sub(r'[^\d.]', '', prices_on_line[-1].replace(',', '').replace(' ', '')) or 0)
                
                p1_dec = len(prices_on_line[-2].split('.')[-1]) if '.' in prices_on_line[-2] else 0
                p2_dec = len(prices_on_line[-1].split('.')[-1]) if '.' in prices_on_line[-1] else 0
                
                # Pokud má jedna cena 4 desetinná místa, je to pravděpodobně jednotková cena bez DPH
                if p1_dec > p2_dec and p1_val < p2_val:
                     item["cena_za_kus"] = {"hodnota": prices_on_line[-2].strip(), "box": []}
                     item["cena_celkem"] = {"hodnota": prices_on_line[-1].strip(), "box": []}
                elif p2_val > p1_val:
                    item["cena_za_kus"] = {"hodnota": prices_on_line[-2].strip(), "box": []}
                    item["cena_celkem"] = {"hodnota": prices_on_line[-1].strip(), "box": []}
                else:
                    # Fallback na pořadí
                    item["cena_za_kus"] = {"hodnota": prices_on_line[-1].strip(), "box": []}
                    item["cena_celkem"] = {"hodnota": prices_on_line[-2].strip(), "box": []}
            elif prices_on_line:
                item["cena_celkem"] = {"hodnota": prices_on_line[0].strip(), "box": []}
            
            data["produkty"].append(item)
            row_index += 1

    return data


def extract_data_with_coords(img_array, mode="heuristic", api_key=None):
    """
    Hlavní orchestrátor extrakce dat z faktury.
    Provádí OCR, volá příslušné modely (Heuristika/Template/AI) a doplňuje souřadnice.
    
    Args:
        img_array (np.ndarray): Obrázek v RGB formátu.
        mode (str): Režim extrakce ("heuristic" nebo "ai").
        api_key (str): Klíč pro AI služby.
        
    Returns:
        tuple: (data_dict, raw_ocr_dict, meta_dict)
    """
    # 1. OCR Fáze: Tesseract převede obrázek na text a geometrická data
    d = pytesseract.image_to_data(img_array, lang='ces', output_type=Output.DICT)
    all_text = " ".join([str(w) for w in d['text'] if str(w).strip()])
    lines = build_lines_from_ocr(d)
    img_shape = img_array.shape

    # 2. AI Fáze (Volitelná): Použití LLM (Gemini/DeepSeek)
    if mode == "ai" and api_key:
        raw_ai_data = extract_with_llm(all_text, api_key)
        if raw_ai_data:
            # Back-mapping: Propojíme JSON hodnoty z LLM se souřadnicemi z OCR
            data = map_json_to_boxes(raw_ai_data, d)
            llm_name = "OpenRouter" if api_key.startswith("sk-or-") else "Gemini"
            data["metadata"] = {"typ_dokumentu": classify_doc(all_text), "stranky": 1, "rezim": f"AI ({llm_name})"}
            return data, d, {"mode": "ai", "ico": None}

    # 3. Detekce IČO: Zjistíme, zda známe dodavatele
    ico = find_ico_in_lines(lines)

    # 4. Šablonová Fáze: Pokud máme uloženou šablonu (100% přesnost)
    if mode != "ai" and ico and store.load(ico):
        template = store.load(ico)
        template_result = apply_template(template, lines, img_shape)
        structured = {
            "metadata": {"typ_dokumentu": classify_doc(all_text), "stranky": 1, "rezim": "TEMPLATE", "ico_key": ico},
            "sablona_ico": ico,
            "sablona_verze": template.get("meta", {}).get("version", 1),
        }
        structured.update(template_result)
        return structured, d, {"mode": "template", "ico": ico}

    # 5. Heuristická Fáze (Fallback): Použití sady pravidel a fuzzy matchingu
    data = heuristic_extract(lines, img_shape, all_text)
    data["metadata"]["rezim"] = "HEURISTIKA"

    # Příprava kandidáta pro novou šablonu (pro uložení uživatelem)
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
