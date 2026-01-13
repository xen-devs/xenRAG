import os
from xenrag.graph.graph import build_graph

def visualize():
    print("Building graph...")
    app = build_graph()
    
    print("Generating Mermaid PNG...")
    try:
        png_data = app.get_graph().draw_mermaid_png()
        output_path = "xenrag_graph.png"
        with open(output_path, "wb") as f:
            f.write(png_data)
        print(f"Graph saved to {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Failed to generate graph: {e}")

if __name__ == "__main__":
    visualize()
