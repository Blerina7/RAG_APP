"""
CLI Interface për RAG App
Përdorim:
  python cli.py ingest <skedar>
  python cli.py query "<pyetja>"
  python cli.py stats
  python cli.py delete <emri_burimit>
"""

import sys
import json
import argparse
from pathlib import Path

# Shto src në path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from rag_engine import RAGApp

# ── Konfigurim ──────────────────────────────────
OLLAMA_MODEL = "llama3"        # ndrysho në "mistral", "phi3", etj.
OLLAMA_URL   = "http://localhost:11434"
CHROMA_DIR   = "./chroma_db"


def print_header():
    print("\n" + "═" * 55)
    print("  🔍  RAG App  |  Ollama + ChromaDB")
    print("═" * 55)


def cmd_ingest(app: RAGApp, filepath: str):
    """Ngarkon dhe indekson një dokument."""
    print(f"\n📄 Duke ngarkuar: {filepath}")
    if not Path(filepath).exists():
        print(f"❌ Skedari nuk u gjet: {filepath}")
        return
    result = app.ingest(filepath)
    if result["success"]:
        print(f"✅ U indeksua: {result['source']}")
        print(f"   • Copa (chunks): {result['chunks']}")
        print(f"   • Karaktere:     {result['chars']}")
    else:
        print(f"❌ Gabim: {result['error']}")


def cmd_query(app: RAGApp, question: str):
    """Bën query dhe shfaq përgjigjen."""
    print(f"\n❓ Pyetja: {question}")
    print("⏳ Duke kërkuar...")

    result = app.query(question)

    print("\n" + "─" * 55)
    print("💡 PËRGJIGJA:")
    print("─" * 55)
    print(result["answer"])

    if result["sources"]:
        print("\n📚 Burimet e përdorura:")
        for s in result["sources"]:
            bar = "█" * int(s["score"] * 10)
            print(f"   • {s['source']}  [{bar:<10}] {s['score']:.2f}")
    print("─" * 55)


def cmd_stats(app: RAGApp):
    """Shfaq statistika."""
    stats = app.get_stats()
    print("\n📊 STATISTIKA:")
    print(f"   • Gjithsej chunks:  {stats['total_chunks']}")
    print(f"   • Ollama aktiv:     {'✅ Po' if stats['ollama_available'] else '❌ Jo'}")
    print(f"   • Model:            {stats['ollama_model']}")

    sources = stats.get("sources", [])
    if sources:
        print(f"\n📁 Dokumentet e indeksuara ({len(sources)}):")
        for s in sources:
            print(f"   • {s}")
    else:
        print("\n   ℹ️  Asnjë dokument i indeksuar ende.")


def cmd_delete(app: RAGApp, source_name: str):
    """Fshin një burim nga index."""
    print(f"\n🗑️  Duke fshirë: {source_name}")
    deleted = app.delete_source(source_name)
    if deleted:
        print(f"✅ U fshinë {deleted} chunks.")
    else:
        print(f"⚠️  Burimi '{source_name}' nuk u gjet.")


def interactive_mode(app: RAGApp):
    """Mënyra interaktive - chat me dokumente."""
    print_header()
    print("\n🎯 Mënyra Interaktive")
    print("   Komandat: /ingest <skedar>, /stats, /delete <burim>, /quit\n")

    while True:
        try:
            user_input = input("💬 Ti: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Mirupafshim!")
            break

        if not user_input:
            continue
        elif user_input.startswith("/ingest "):
            filepath = user_input[8:].strip()
            cmd_ingest(app, filepath)
        elif user_input == "/stats":
            cmd_stats(app)
        elif user_input.startswith("/delete "):
            source = user_input[8:].strip()
            cmd_delete(app, source)
        elif user_input in ["/quit", "/exit", "/q"]:
            print("\n👋 Mirupafshim!")
            break
        else:
            cmd_query(app, user_input)


def main():
    parser = argparse.ArgumentParser(
        description="🔍 RAG App - Pyet dokumentat tuaja",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Shembuj:
  python cli.py ingest docs/raporti.pdf
  python cli.py query "Çfarë thotë raporti për financat?"
  python cli.py stats
  python cli.py          # mënyra interaktive
        """
    )
    parser.add_argument("command", nargs="?",
                        choices=["ingest", "query", "stats", "delete"],
                        help="Komanda")
    parser.add_argument("argument", nargs="?", help="Argumenti (skedar ose pyetja)")
    parser.add_argument("--model", default=OLLAMA_MODEL,
                        help=f"Modeli Ollama (default: {OLLAMA_MODEL})")
    parser.add_argument("--ollama-url", default=OLLAMA_URL,
                        help=f"URL e Ollama (default: {OLLAMA_URL})")
    parser.add_argument("--db-dir", default=CHROMA_DIR,
                        help=f"Drejtoria e ChromaDB (default: {CHROMA_DIR})")

    args = parser.parse_args()

    # Inicializo RAG App
    app = RAGApp(
        ollama_model=args.model,
        persist_dir=args.db_dir,
        ollama_url=args.ollama_url
    )

    print_header()

    if args.command is None:
        interactive_mode(app)
    elif args.command == "ingest":
        if not args.argument:
            print("❌ Specifiko skedarin: python cli.py ingest <skedar>")
            sys.exit(1)
        cmd_ingest(app, args.argument)
    elif args.command == "query":
        if not args.argument:
            print("❌ Specifiko pyetjen: python cli.py query '<pyetja>'")
            sys.exit(1)
        cmd_query(app, args.argument)
    elif args.command == "stats":
        cmd_stats(app)
    elif args.command == "delete":
        if not args.argument:
            print("❌ Specifiko burimin: python cli.py delete <emri>")
            sys.exit(1)
        cmd_delete(app, args.argument)


if __name__ == "__main__":
    main()
