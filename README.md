# 📄 PDF Analyzer: Enterprise Grade Invoice Extraction

A sophisticated Python-based pipeline for automated invoice processing, combining **Ultra-fast Local Heuristics** with **Advanced Generative AI (Gemini)**.

---

## 🚀 The Core Pipeline
Our architectural approach ensures 100% data reliability by prioritizing local deterministic logic before falling back to expensive AI models.

1.  **OCR Layer**: High-resolution text extraction using Tesseract OCR.
2.  **Classification**: Real-time document typing (Invoice, Offer, Email).
3.  **Template Engine**: Instant recognition based on Seller IČO (0ms latency for known issuers).
4.  **Heuristic Engine**: Fuzzy label matching and spatial anchor search for unknown layouts.
5.  **AI Failover**: Smart integration with Google Gemini 1.5/2.0 for complex, multi-page international invoices.
6.  **Back-mapping**: Synchronizing extracted AI text with OCR coordinates for pixel-perfect UI visualization.

---

## 🧠 Smart Engines

### 🔍 Heuristic Engine (Python/OpenCV)
*   **Fuzzy Logic**: Matches variations of "Odběratel", "Dodavatel", "IČO" etc., across different languages and encodings.
*   **Spatial Search**: Uses geometric "anchors" to find values near labels (e.g., finding the price strictly to the right or below the "Total" label).
*   **Regex Clusters**: Optimized patterns for Czech and European price formats, including high-precision unit prices (4 decimals).

### 🤖 AI Brain (Google Gemini)
*   **Model Fallback**: Automatically cycles through `gemini-1.5-flash`, `gemini-flash-latest`, and `gemini-2.0-flash` to bypass quota or availability issues.
*   **Contextual Understanding**: Extracts semantics even from distorted or non-standard tables where traditional regex fails.
*   **Sequence Mapping**: Custom algorithm that reconstructs full-sentence bounding boxes from individual OCR words.

---

## 📦 Enterprise JSON Schema
We transform raw text into developer-friendly, normalized data structures:

```json
{
  "prodavajici": {
    "nazev": "Electroshop s.r.o.",
    "ico": "23132456454",
    "adresa": { "ulice": "Kobližná", "mesto": "Jablonec", "psc": "12345" }
  },
  "faktura": {
    "cislo_faktury": "20241014",
    "datum_vystaveni": "2025-09-03"
  },
  "celkem": {
    "hodnota": 1205.0,
    "mena": "USD",
    "raw": "$1,205.00"
  },
  "metadata": {
    "rezim": "AI (Gemini)",
    "typ_dokumentu": "FAKTURA"
  }
}
```

---

## 🎨 Professional UI (Streamlit)
*   **Live Preview**: Real-time PDF rendering with page navigation.
*   **Visual Highlights**: Red bounding boxes for extracted values + Blue regions for detected tables.
*   **Template Manager**: Save new extraction rules directly from the UI with one click.

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.9+
- Tesseract OCR (installed in `C:/Program Files/Tesseract-OCR`)

### 2. Installation
```bash
pip install -r requirements.txt
```

### 3. Launch
```bash
streamlit run app.py
```

---

## 📂 Examples
The project includes a suite of test invoices in the `@examples` folder, covering:
- ✅ Classic Czech layouts (TechMorava)
- ✅ International English invoices (Blue Horizon)
- ✅ Modern minimalist designs (Apple Style)
- ✅ Edge cases with mixed currencies (EUR/CZK)

---
*Built with ❤️ for rapid invoice automation.*
