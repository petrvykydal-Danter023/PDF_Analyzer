import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import json

# --- MODULY PRO EXTRAKCI A SPRÁVU ŠABLON ---
from extraction_logic import extract_data_with_coords
from template_engine import TemplateStore
import pypdfium2 as pdfium

st.set_page_config(layout="wide", page_title="Extrakce Faktur – Auto-Template Engine")
store = TemplateStore()

# --- HLAVNÍ NADPIS A POPIS SYSTÉMU ---
st.title("📄 Extrakce Faktur – Auto-Template Engine")
st.markdown("""
**Jak to funguje:**
- Nahrajte fakturu → Tesseract přečte text + souřadnice (bounding boxy).
- Pokud je **IČO firmy** v databázi → použije se uložená YAML šablona (100 % deterministicky).
- Pokud ne → provede se heuristická extrakce + systém navrhne novou šablonu.
- Šablonu si **zkontrolujete v UI** a jedním klikem uložíte → příští faktura stejné firmy = perfektní výsledek.
""")

# --- DATABÁZE ŠABLON (PŘEHLED) ---
known = store.list_templates()
if known:
    st.sidebar.success(f"📁 Šablony v databázi: **{len(known)}** firem")
    st.sidebar.caption("IČO: " + ", ".join(known))
else:
    st.sidebar.info("Zatím žádné šablony. Nahrajte první fakturu!")

# --- NASTAVENÍ EXTRAKCE (REŽIMY) ---
st.sidebar.subheader("⚙️ Nastavení extrakce")
extraction_mode = st.sidebar.radio(
    "Metoda extrakce:",
    ["Heuristika / Šablony", "AI (LLM)"],
    index=0,
    help="AI mód (Gemini/OpenRouter) je vhodnější pro atypické nebo anglické dokumenty."
)

ai_key = None
if extraction_mode == "AI (LLM)":
    # API klíč pro LLM modely (podporuje Gemini i OpenRouter)
    ai_key = st.sidebar.text_input("API Key", type="password", help="Klíč začínající 'sk-or-' použije OpenRouter.")
    if not ai_key:
        st.sidebar.warning("⚠️ Pro AI mód je nutný API klíč.")

uploaded_file = st.file_uploader("Nahrajte fakturu (PNG / JPG / PDF)", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    # 1. Převod dokumentu na manipulovatelný obrázek (Bitmapa)
    if uploaded_file.name.lower().endswith('.pdf'):
        # PDF se převede na renderovanou bitmapu pomocí pypdfium2 (pouze 1. strana)
        pdf = pdfium.PdfDocument(uploaded_file.read())
        page = pdf.get_page(0)
        # Používáme škálování 3x pro dosažení vysokého rozlišení (klíčové pro přesné OCR)
        image = page.render(scale=3).to_pil()
    else:
        # Přímé načtení rastrového obrázku (PNG/JPG)
        image = Image.open(uploaded_file)

    # --- HLAVNÍ EXTRAKČNÍ PIPELINE ---
    # Převod volby z UI na interní parametr (ai vs heuristic)
    mode_val = "ai" if extraction_mode == "AI (LLM)" else "heuristic"
    
    with st.spinner('🔍 Analyzuji dokument...'):
        try:
            # Volání orchestrátoru, který provede OCR a sémantickou analýzu
            data, raw_ocr, meta = extract_data_with_coords(img_array, mode=mode_val, api_key=ai_key)
            success = True
        except Exception as e:
            st.error(f"Chyba při zpracování: {e}")
            import traceback
            st.code(traceback.format_exc())
            success = False

    if success:
        mode = meta.get("mode", "heuristic")
        ico = meta.get("ico")

        if mode == "template":
            st.success(f"✅ Použita uložená šablona pro IČO **{ico}** – deterministická extrakce!")
        else:
            st.warning(f"⚠️ Firma {'(IČO: ' + ico + ')' if ico else '(IČO nenalezeno)'} nemá uloženou šablonu – použita heuristika.")

        col1, col2 = st.columns([1, 1])

        # ---- Levý sloupec: Vizualizace (Prostorová kontrola) ----
        with col1:
            st.subheader("🖼️ Vizualizace (Bounding Boxy)")
            show_all = st.checkbox("Zobrazit všechna detekovaná pole (žlutě/oranžově)", value=True)
            draw = ImageDraw.Draw(image)

            # Funkce pro rekurzivní vykreslení boxů všech extrahovaných polí
            def draw_boxes(obj, color="red"):
                if isinstance(obj, dict):
                    if "box" in obj and obj["box"]:
                        b = obj["box"]
                        if b and len(b) == 4:
                            # Vykreslení obdélníku kolem hodnoty
                            draw.rectangle([b[0], b[1], b[0]+b[2], b[1]+b[3]], outline=color, width=3)
                            # Zobrazení náhledu textu nad boxem
                            val = str(obj.get("hodnota", ""))[:30]
                            draw.text((b[0], max(0, b[1]-15)), val, fill=color)
                    for k, v in obj.items():
                        if k != "metadata":
                            draw_boxes(v, color)
                elif isinstance(obj, list):
                    for item in obj:
                        draw_boxes(item, color)

            # 1. Vizualizace všech OCR kotev (KV-pairs) - užitečné pro ladění šablon
            if show_all and meta.get("kv_pairs"):
                for kv in meta["kv_pairs"]:
                    lb = kv["label_box"]
                    vb = kv["value_box"]
                    # Žlutá = Štítek (Label), Oranžová = Hodnota (Value)
                    draw.rectangle([lb[0], lb[1], lb[0]+lb[2], lb[1]+lb[3]], outline="yellow", width=2)
                    draw.rectangle([vb[0], vb[1], vb[0]+vb[2], vb[1]+vb[3]], outline="orange", width=2)

            # 2. Úspěšně extrahovaná data (zobrazeno červeně)
            draw_boxes(data, color="red")

            # 3. Zvýraznění tabulkového regionu detekovaného OpenCV (modře)
            if meta.get("table_box"):
                tb = meta["table_box"]
                draw.rectangle([tb[0], tb[1], tb[0]+tb[2], tb[1]+tb[3]], outline="blue", width=4)
                draw.text((tb[0], tb[1]-20), "TABLE REGION (OpenCV)", fill="blue")

            # Zobrazení výsledného obrázku s překryvem
            st.image(image, use_column_width=True)

        # ---- Pravý sloupec: JSON + Human Review ----
        # ---- Pravý sloupec: JSON výsledky a Human-in-the-Loop review ----
        with col2:
            st.subheader("📊 Výsledný JSON")
            st.json(data)

            # RAW OCR výstup pro pokročilé uživatele a ladění
            with st.expander("🔎 Kontrola surového OCR textu (Tesseract)"):
                raw_text = " ".join([str(w) for w in raw_ocr['text'] if str(w).strip()])
                st.text_area("To, co systém vidí jako text:", raw_text, height=120)

            # ======================================================
            # HUMAN-IN-THE-LOOP: Uložit šablonu (Doučování systému)
            # ======================================================
            if mode == "heuristic" and meta.get("candidate_template"):
                st.divider()
                st.subheader("🧠 Uložit šablonu pro tuto firmu")
                st.info(
                    "Systém automaticky detekoval strukturu dokladu. "
                    "Zkontrolujte vytažená pole níže. "
                    "Po uložení bude příští faktura od stejné firmy zpracována automaticky."
                )

                candidate = meta["candidate_template"]
                ico_input = st.text_input(
                    "IČO firmy (klíč šablony)",
                    value=ico or "",
                    help="IČO slouží jako klíč šablony. Musí být 8 číslic pro CZ firmy."
                )
                issuer_input = st.text_input(
                    "Název firmy (pro přehled)",
                    value=candidate.get("meta", {}).get("issuer", "")
                )

                # Detekované KV páry
                kv_pairs = meta.get("kv_pairs", [])
                st.markdown("**Detekované klíč-hodnota páry:**")

                selected_fields = {}
                if kv_pairs:
                    for idx, kv in enumerate(kv_pairs[:20]):  # Max 20 polí
                        cols = st.columns([2, 3, 1])
                        with cols[0]:
                            st.caption(f"🏷 {kv['label']}")
                        with cols[1]:
                            val_input = st.text_input(
                                f"Hodnota ({kv['label']})",
                                value=kv['value'],
                                key=f"kv_{kv['label']}_{idx}",
                                label_visibility="collapsed"
                            )
                        with cols[2]:
                            include = st.checkbox("✓", value=True, key=f"inc_{kv['label']}_{idx}")
                        if include:
                            selected_fields[f"{kv['label']}_{idx}"] = {**kv, "value": val_input}

                # Tabulková oblast
                if meta.get("table_box"):
                    st.markdown("**OpenCV detekoval tabulkovou oblast** (modrý rámeček v obrázku)")

                # Uložit šablonu
                if st.button("💾 Uložit šablonu", type="primary"):
                    if not ico_input or (len(re.sub(r'\D', '', ico_input)) not in [6, 8, 9]):
                        st.error("Zadejte platné IČO (6–9 číslic)")
                    else:
                        # Aktualizuj šablonu s vybranými poli
                        candidate["meta"]["ico"] = ico_input
                        candidate["meta"]["issuer"] = issuer_input
                        # Filtruj fieldy jen na vybraná
                        filtered_fields = {}
                        for kv in selected_fields.values():
                            label_key = re.sub(r"[^a-z0-9_]", "_", kv['label'].lower())
                            if label_key in candidate["fields"]:
                                filtered_fields[label_key] = candidate["fields"][label_key]
                                filtered_fields[label_key]["value_snapshot"] = kv["value"]
                        candidate["fields"] = filtered_fields
                        store.save(ico_input, candidate)
                        st.success(f"✅ Šablona pro IČO **{ico_input}** ({issuer_input}) uložena! Příští faktura bude zpracována automaticky.")
                        st.balloons()

            elif mode == "template":
                st.divider()
                with st.expander("⚙️ Správa šablony"):
                    st.caption(f"Šablona pro IČO **{ico}** je aktivní.")
                    if st.button("🗑️ Smazat šablonu (vrátit na heuristiku)"):
                        import os
                        path = store._path(ico)
                        if os.path.exists(path):
                            os.remove(path)
                            st.success("Šablona smazána. Při příštím uploadu se použije heuristika.")

import re
