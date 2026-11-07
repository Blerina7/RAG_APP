import sys
from src.rag_engine import RAGEngine

def main():
    print("Po karikojme modelet lokale. Kjo mund te marre pak sekonda...")
    engine = RAGEngine()
    print("Sistemi RAG eshte gati! Shkruaj /quit per te dale.")
    
    while True:
        try:
            user_input = input("\n[RAG App]> ")
            if not user_input.strip():
                continue
                
            parts = user_input.split(" ", 1)
            command = parts[0]
            
            if command == "/quit":
                print("Po mbyllet...")
                break
            elif command == "/ingest":
                if len(parts) < 2:
                    print("Perdorimi: /ingest rruga/drejt/skedarit.pdf")
                    continue
                file_path = parts[1]
                print(f"Po lexojme {file_path} dhe po ruajme vektoret...")
                chunks = engine.ingest_pdf(file_path)
                print(f"Sukses! Dokumenti u nda ne {chunks} copa dhe u ruajt ne ChromaDB.")
            else:
                # Nese nuk eshte komande me '/', trajtohet si pyetje per dokumentin
                print("Duke kerkuar ne dokumente dhe gjeneruar pergjigjen...")
                answer = engine.query(user_input)
                print(f"\nPergjigjja: {answer}")
                
        except KeyboardInterrupt:
            print("\nPo mbyllet...")
            break
        except Exception as e:
            print(f"Ndodhi nje gabim: {e}")

if __name__ == "__main__":
    main()