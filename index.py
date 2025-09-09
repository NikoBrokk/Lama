# index.py
# Formål: ta chunks fra ingest.py → gjør dem til LlamaIndex Documents → bygg VectorStoreIndex → persistér til disk.

from __future__ import annotations                      # # gjør type hints mer fleksible
import os                                               # # lese miljøvariabler (OPENAI_API_KEY) + stier
import argparse                                         # # enkel CLI for terminal-opsjoner
from typing import List, Dict                           # # type hints
from ingest import load_pdf, chunk_text                 # # gjenbruk vår egen ingest-pipeline
from llama_index.core import Document, VectorStoreIndex # # LlamaIndex grunnklasser vi trenger
from llama_index.core import Settings                   # # global innstilling for embeddings/LLM
from llama_index.embeddings.openai import OpenAIEmbedding  # # OpenAI embeddings (text-embedding-3-*)
# Merk: vi setter ikke LLM her ennå – kun embeddings for å bygge index

def _to_documents(chunks: List[Dict]) -> List[Document]:
    """
    Konverterer våre dict-chunks til LlamaIndex Document-objekter.
    Hver Document har text + metadata (page, chunk_id, source).
    """
    docs: List[Document] = []                           # # samler opp dokumenter
    for c in chunks:                                    # # iterer gjennom alle chunks
        text = c["text"]                                # # selve tekstinnholdet
        metadata = {                                    # # enkel metadata for sporbarhet
            "page": c.get("page"),
            "chunk_id": c.get("chunk_id"),
            "source": c.get("source", "")
        }
        docs.append(Document(text=text, metadata=metadata))  # # bygg Document-objektet
    return docs                                         # # returner liste av Document

def build_index_from_pdf(
    pdf_path: str,
    chunk_size: int = 800,
    overlap: int = 120,
    persist_dir: str = "./storage"
) -> VectorStoreIndex:
    """
    Leser PDF → lager chunks → bygger VectorStoreIndex → persisterer til disk.
    Returnerer index-objektet (kan brukes direkte hvis ønskelig).
    """
    pages = load_pdf(pdf_path)                          # # les tekst per side via pypdf
    chunks = chunk_text(pages, chunk_size, overlap, source_path=pdf_path)  # # vår egen chunker
    docs = _to_documents(chunks)                        # # konverter til LlamaIndex Document

    # # Bygg selve indexen (dette trengs embeddings – vi satte Settings.embed_model globalt i main())
    index = VectorStoreIndex.from_documents(docs)       # # lager vektorindeks (embeddings bak kulissene)

    # # Lagre index til disk slik at query.py kan laste uten å re-indeksere
    os.makedirs(persist_dir, exist_ok=True)             # # sørg for at mappa finnes
    index.storage_context.persist(persist_dir=persist_dir)  # # skriv ut lagringsartefakter
    return index                                        # # gi tilbake index for evt. on-the-fly spørring senere

def _find_first_pdf(data_dir: str = "./data") -> str:
    """
    Liten helper: finn første pdf i ./data hvis bruker ikke spesifiserer --path
    """
    import glob, os                                     # # lokal import (smått verktøy)
    candidates = sorted(glob.glob(os.path.join(data_dir, "*.pdf")))  # # alle PDFer, sortert
    if not candidates:
        raise FileNotFoundError(f"Ingen PDF funnet i {data_dir}.")
    return candidates[0]                                # # ta første for determinisme

def main():
    parser = argparse.ArgumentParser(description="Bygg LlamaIndex VectorStoreIndex fra en PDF.")
    parser.add_argument("--path", type=str, default=None, help="Sti til PDF (default: første i ./data)")
    parser.add_argument("--chunk-size", type=int, default=800, help="Antall tegn per chunk")
    parser.add_argument("--overlap", type=int, default=120, help="Overlapp mellom chunks i tegn")
    parser.add_argument("--persist-dir", type=str, default="./storage", help="Mappe for persistering av index")
    args = parser.parse_args()                          # # parse CLI-argumenter

    # # Sjekk at vi har API-nøkkel til embeddings (OpenAI). Vi stopper tidlig hvis ikke.
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "Manglende OPENAI_API_KEY. Sett den i miljøet: "
            "export OPENAI_API_KEY='sk-...'  (mac/Linux) eller via shell/profil."
        )

    # # Sett embeddings-modellen globalt for LlamaIndex (billig og bra default)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")  # # rask/billig; kan byttes til -large

    pdf_path = args.path or _find_first_pdf()          # # finn filsti (brukerspesifisert eller første i ./data)
    print(f"Bygger index fra: {pdf_path}")             # # liten statuslinje

    index = build_index_from_pdf(                       # # kall hoved-funksjonen
        pdf_path=pdf_path,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        persist_dir=args.persist_dir
    )

    # # Litt nyttig statistikk til terminalen
    # # (VectorStoreIndex i seg selv eksponerer ikke antall dokumenter direkte – vi bruker storage_context)
    docstore = index.storage_context.docstore          # # hent dokumentlageret
    print(f"Antall lagrede noder: {len(docstore.docs)}")  # # grovt mål på hvor mye som ble indeksert
    print(f"Index persistert til: {args.persist_dir}")    # # hvor query.py kan finne den igjen

if __name__ == "__main__":
    main()                                             # # kjør main når scriptet kjøres direkte
