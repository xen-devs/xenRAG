import asyncio
from xenrag.graph.graph import build_graph

async def main():
    print("Initializing XenRAG Graph...")
    try:
        app = build_graph()
    except Exception as e:
        print(f"Failed to build graph: {e}")
        return

    print("Graph initialized. Ready for input.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("User: ")
        except EOFError:
            break

        if user_input.lower() in ["exit", "quit"]:
            break
        
        if not user_input.strip():
            continue

        print("Processing...")
        
        inputs = {"input_query": user_input}
        
        try:
            # Invoke the graph
            result = await app.ainvoke(inputs)
            
            print("\n--- Output ---")
            
            intent = result.get("intent")
            if intent:
                print(f"Intent: {intent.type} (Confidence: {intent.confidence:.2f})")
            
            emotion = result.get("emotion")
            if emotion:
                print(f"Emotion: {emotion.type} (Confidence: {emotion.confidence:.2f})")
            
            reasoning = result.get("private_reasoning")
            if reasoning:
                print("\nReasoning Trace:")
                for record in reasoning:
                    print(f"- [{record.step}] {record.summary}")

            answer = result.get("generated_answer")
            if answer:
                print(f"\nAnswer: {answer}")
            
            print("-" * 20 + "\n")

        except Exception as e:
            print(f"Error executing graph: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExiting...")
