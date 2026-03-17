import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import json

from extraction_logic import extract_data_with_coords
from template_engine import TemplateStore, build_lines_from_ocr
import pypdfium2 as pdfium

st.set_page_config(layout="wide", page_title="Extrakce Faktur – Auto-Template Engine")
store = TemplateStore()

st.title("📄 Extrakce Faktur – Auto-Template Engine")
st.markdown("""
**Jak to funguje:**
- Nahrajte fakturu → Tesseract přečte text + souřadnice (bounding boxy).
- Pokud je **IČO firmy** v databázi → použije se uložená YAML šablona (100 % deterministicky).
- Pokud ne → provede se heuristická extrakce + systém navrhne novou šablonu.
- Šablonu si **zkontrolujete v UI** a jedním klikem uložíte → příští faktura stejné firmy = perfektní výsledek.
""")

known = store.list_templates()
if known:
    st.sidebar.success(f"📁 Šablony v databázi: **{len(known)}** firem")
    st.sidebar.caption("IČO: " + ", ".join(known))
else:
    st.sidebar.info("Zatím žádné šablony. Nahrajte první fakturu a uložte šablonu!")

uploaded_file = st.file_uploader("Nahrajte fakturu (PNG / JPG / PDF)", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    # PDF → obrázek (první strana)
    if uploaded_file.name.lower().endswith('.pdf'):
        pdf = pdfium.PdfDocument(uploaded_file.read())
        page = pdf.get_page(0)
        image = page.render(scale=3).to_pil()
    else:
        image = Image.open(uploaded_file)

    img_array = np.array(image.convert('RGB'))

    with st.spinner('🔍 Analyzuji dokument (Tesseract OCR + template engine)...'):
        try:
            data, raw_ocr, meta = extract_data_with_coords(img_array)
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

        # ---- Levý sloupec: Vizualizace ----
        with col1:
            st.subheader("Vizualizace (Bounding Boxy)")
            draw = ImageDraw.Draw(image)

            def draw_boxes(obj, color="red"):
                if isinstance(obj, dict):
                    if "box" in obj and obj["box"]:
                        b = obj["box"]
                        draw.rectangle([b[0], b[1], b[0]+b[2], b[1]+b[3]], outline=color, width=3)
                        val = str(obj.get("hodnota", ""))[:30]
                        draw.text((b[0], max(0, b[1]-15)), val, fill=color)
                    for k, v in obj.items():
                        draw_boxes(v, color)
                elif isinstance(obj, list):
                    for item in obj:
                        draw_boxes(item, color)

            draw_boxes(data, color="red")

            # Tabulkový region (OpenCV)
            if meta.get("table_box"):
                tb = meta["table_box"]
                draw.rectangle([tb[0], tb[1], tb[0]+tb[2], tb[1]+tb[3]], outline="blue", width=4)
                draw.text((tb[0], tb[1]-20), "TABLE REGION (OpenCV)", fill="blue")

            st.image(image, use_column_width=True)

        # ---- Pravý sloupec: JSON + Human Review ----
        with col2:
            st.subheader("Výsledný JSON")
            st.json(data)

            # RAW OCR text pro debugging
            with st.expander("🔎 Raw OCR text"):
                raw_text = " ".join([w for w in raw_ocr['text'] if str(w).strip()])
                st.text_area("Tesseract přečetl:", raw_text, height=120)

            # ======================================================
            # HUMAN-IN-THE-LOOP: Uložit šablonu
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
                    for kv in kv_pairs[:20]:  # Max 20 polí
                        cols = st.columns([2, 3, 1])
                        with cols[0]:
                            st.caption(f"🏷 {kv['label']}")
                        with cols[1]:
                            val_input = st.text_input(
                                f"Hodnota ({kv['label']})",
                                value=kv['value'],
                                key=f"kv_{kv['label']}",
                                label_visibility="collapsed"
                            )
                        with cols[2]:
                            include = st.checkbox("✓", value=True, key=f"inc_{kv['label']}")
                        if include:
                            selected_fields[kv['label']] = {**kv, "value": val_input}

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
