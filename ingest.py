# ingest.py
# Formål: lese en PDF fra ./data, dele opp i mindre tekstbiter (chunks), og printe statistikk.
# Vi holder det enkelt nå (pypdf + ren tekst). Metadata og persistering kan vi legge til i neste steg.

from __future__ import annotations  # # gjør type hints litt mer fleksible (fremtidsvennlig)
import argparse                     # # enkel CLI-parsing i terminal
import os                           # # for filstier og sjekk av fil-eksistens
import glob                         # # for å finne første PDF i ./data hvis path ikke er spesifisert
from typing import List, Dict       # # type hints for bedre lesbarhet
from pypdf import PdfReader         # # biblioteket som faktisk leser PDF-tekst

def load_pdf(path: str) -> List[str]:
    """
    Leser en PDF og returnerer en liste med tekst per side.
    """
    if not os.path.exists(path):                    # # verifiser at fila finnes
        raise FileNotFoundError(f"Fant ikke fil: {path}")
    if not path.lower().endswith(".pdf"):          # # enkel sjekk: vi forventer .pdf
        raise ValueError("Fil må være en .pdf")

    reader = PdfReader(path)                        # # åpner PDF-en
    pages_text: List[str] = []                      # # her samler vi tekst per side
    for page in reader.pages:                       # # iterer gjennom sidene
        text = page.extract_text() or ""            # # hent ut ren tekst (kan være None → tom streng)
        pages_text.append(_clean_text(text))        # # normaliser whitespace for enklere chunking
    return pages_text                               # # resultat: liste[str] med én streng per side

def _clean_text(text: str) -> str:
    """
    Enkel normalisering av tekst: strip + komprimer whitespace.
    """
    return " ".join(text.strip().split())          # # fjerner linjeskift/dobbel-spaces → én space

def chunk_text(
    pages: List[str],
    chunk_size: int = 800,
    overlap: int = 120,
    source_path: str | None = None
) -> List[Dict]:
    """
    Deler opp tekst per side i overlappende biter (chunks).
    Returnerer liste av dicts med enkel metadata (side, id, tekst).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size må være > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap må være >= 0 og < chunk_size")

    chunks: List[Dict] = []                         # # her samler vi alle chunk-objektene
    step = chunk_size - overlap                     # # hvor mye vi hopper for hver chunk
    for i, page_text in enumerate(pages, start=1):  # # i = sidenummer (1-basert)
        n = 0                                       # # løpende chunk-indeks innen siden
        idx = 0                                     # # startposisjon i teksten
        while idx < len(page_text):                 # # lag nye chunks til vi er gjennom siden
            part = page_text[idx: idx + chunk_size] # # ta ut en bit på chunk_size tegn
            if not part:                            # # sikkerhetsnett
                break
            chunk = {
                "text": part,                       # # selve tekstbiten
                "page": i,                          # # hvor i PDF-en den kom fra
                "chunk_id": f"{i}-{n}",             # # unik id per side og rekkefølge
                "source": source_path or ""         # # filsti (kan være tom i første omgang)
            }
            chunks.append(chunk)                    # # legg til i resultat
            n += 1                                  # # øk chunk-indeksen
            idx += step                             # # hopp frem med step (chunk_size - overlap)
    return chunks                                   # # ferdig: liste[dict] med chunks + enkel metadata

def _find_first_pdf(data_dir: str = "./data") -> str:
    """
    Hjelpefunksjon: finn første PDF i data-mappa.
    """
    pattern = os.path.join(data_dir, "*.pdf")       # # f.eks. "./data/*.pdf"
    candidates = sorted(glob.glob(pattern))         # # finn alle PDF-er og sorter for determinisme
    if not candidates:
        raise FileNotFoundError(f"Fant ingen PDF i {data_dir}. Legg en fil der.")
    return candidates[0]                            # # returner første treff

def main():
    parser = argparse.ArgumentParser(description="Ingest: les PDF → chunks")
    parser.add_argument("--path", type=str, default=None, help="Sti til PDF (default: første i ./data)")
    parser.add_argument("--chunk-size", type=int, default=800, help="Antall tegn per chunk")
    parser.add_argument("--overlap", type=int, default=120, help="Overlapp mellom chunks i tegn")
    parser.add_argument("--preview", type=int, default=1, help="Hvor mange chunks å printe som eksempel")
    args = parser.parse_args()                      # # parse CLI-argumenter fra terminal

    pdf_path = args.path or _find_first_pdf()       # # bruk oppgitt sti, ellers første PDF i ./data
    pages = load_pdf(pdf_path)                      # # les tekst per side
    chunks = chunk_text(pages, args.chunk_size, args.overlap, source_path=pdf_path)  # # lag chunks

    print(f"Fil: {pdf_path}")                       # # oppsummering til terminalen
    print(f"Antall sider: {len(pages)}")
    print(f"Antall chunks: {len(chunks)} (chunk_size={args.chunk_size}, overlap={args.overlap})")

    # # Vis et lite utdrag for sanity-check (ikke hele – vi holder det kort)
    for c in chunks[: max(0, args.preview)]:
        preview = c['text'][:200] + ("..." if len(c['text']) > 200 else "")
        print(f"\nChunk {c['chunk_id']} (side {c['page']}):\n{preview}")

if __name__ == "__main__":
    main()                                          # # start programmet når filen kjøres direkte
