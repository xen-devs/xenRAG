"""
Interactive CLI for XenRAG
"""

import asyncio
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich import box

console = Console()

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "dim": "\033[2m",
}


def print_header():
    """Print welcome header."""
    console.print(Panel.fit(
        "[bold cyan]XenRAG[/bold cyan] - Explainable RAG Chatbot",
        border_style="cyan"
    ))
    console.print()


def print_result(result: dict):
    """Pretty print the graph result."""
    console.print()
    
    # Intent & Emotion
    intent = result.get("intent")
    emotion = result.get("emotion")
    
    if intent or emotion:
        info_table = Table(show_header=False, box=box.SIMPLE, padding=(0, 2))
        info_table.add_column("Field", style="dim")
        info_table.add_column("Value")
        
        if intent:
            confidence_color = "green" if intent.confidence >= 0.7 else "yellow" if intent.confidence >= 0.5 else "red"
            info_table.add_row("Intent", f"[bold]{intent.type}[/bold] [{confidence_color}]({intent.confidence:.0%})[/{confidence_color}]")
        
        if emotion:
            info_table.add_row("Emotion", f"[bold]{emotion.type}[/bold] ({emotion.confidence:.0%})")
        
        console.print(info_table)
    
    # Check if blocked by guardrails
    if result.get("is_blocked"):
        console.print()
        console.print(Panel(
            f"[red]{result.get('blocked_reason', 'Request blocked by safety filters.')}[/red]",
            title="[bold red]Blocked[/bold red]",
            border_style="red"
        ))
        return
    
    # Check if clarification is needed
    if result.get("needs_clarification") and result.get("clarification_message"):
        console.print()
        console.print(Panel(
            f"[yellow]{result['clarification_message']}[/yellow]",
            title="[bold yellow]Clarification Needed[/bold yellow]",
            border_style="yellow"
        ))
        if result.get("clarification_reason"):
            console.print(f"  [dim]Reason: {result['clarification_reason']}[/dim]")
        return
    
    # Main Answer
    answer = result.get("generated_answer")
    if answer:
        console.print()
        console.print(Panel(
            Markdown(answer),
            title="[bold green]Response[/bold green]",
            border_style="green"
        ))
    
    # Explanations
    explanations = result.get("explanations", [])
    if explanations:
        console.print()
        console.print("[bold magenta]Explanation[/bold magenta]")
        for i, exp in enumerate(explanations):
            console.print(f"  • Reasoning: [cyan]{exp.reasoning_type}[/cyan]")
            console.print(f"  • Confidence: [{'green' if exp.confidence >= 0.7 else 'yellow'}]{exp.confidence:.0%}[/]")
            if exp.evidence_ids:
                console.print(f"  • Sources: {', '.join(exp.evidence_ids[:3])}{'...' if len(exp.evidence_ids) > 3 else ''}")
            if exp.limitations:
                console.print(f"  • [dim]Limitations: {exp.limitations}[/dim]")
    
    # Reasoning Trace
    reasoning = result.get("private_reasoning", [])
    if reasoning:
        console.print()
        console.print("[dim]─── Reasoning Trace ───[/dim]")
        for record in reasoning:
            step = record.step if hasattr(record, 'step') else record.get('step', 'Unknown')
            summary = record.summary if hasattr(record, 'summary') else record.get('summary', '')
            confidence = record.confidence if hasattr(record, 'confidence') else record.get('confidence', 0)
            conf_color = "green" if confidence >= 0.7 else "yellow" if confidence >= 0.5 else "red"
            console.print(f"  [{conf_color}]●[/{conf_color}] [bold]{step}[/bold]: {summary}")


def print_help():
    """Print help message."""
    console.print(Panel(
        "[bold]Commands:[/bold]\n"
        "  [cyan]exit[/cyan], [cyan]quit[/cyan]  - Exit the chatbot\n"
        "  [cyan]help[/cyan]        - Show this help message\n"
        "  [cyan]clear[/cyan]       - Clear the screen\n\n"
        "[bold]Example Queries:[/bold]\n"
        "  • What are the main complaints about battery life?\n"
        "  • Summarize customer feedback on the new features\n"
        "  • Why are customers frustrated with shipping?\n"
        "  • Compare reviews about price vs quality",
        title="[bold]Help[/bold]",
        border_style="blue"
    ))


async def main():
    """Main CLI loop."""
    print_header()
    
    console.print("[dim]Initializing XenRAG Graph...[/dim]")
    
    try:
        from xenrag.graph.graph import build_graph
        app = build_graph()
        console.print("[green]OK[/green] Graph initialized successfully!\n")
    except Exception as e:
        console.print(f"[red]ERROR: Failed to build graph:[/red] {e}")
        return
    
    console.print("Type [cyan]help[/cyan] for commands, or start asking questions.\n")
    
    while True:
        try:
            console.print("[bold cyan]You:[/bold cyan] ", end="")
            user_input = input()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break
        
        # Handle special commands
        cmd = user_input.strip().lower()
        
        if cmd in ["exit", "quit", "q"]:
            console.print("[dim]Goodbye![/dim]")
            break
        
        if cmd == "help":
            print_help()
            continue
        
        if cmd == "clear":
            console.clear()
            print_header()
            continue
        
        if not user_input.strip():
            continue
        
        # Process query
        console.print("[dim]Processing...[/dim]")
        
        inputs = {"input_query": user_input}
        
        try:
            result = await app.ainvoke(inputs)
            print_result(result)
            console.print()
            
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            import traceback
            traceback.print_exc()


async def simple_main():
    """Fallback simple CLI without rich (in case rich is not available)."""
    print("\n" + "=" * 50)
    print("  XenRAG - Explainable RAG Chatbot")
    print("=" * 50 + "\n")
    
    print("Initializing XenRAG Graph...")
    
    try:
        from xenrag.graph.graph import build_graph
        app = build_graph()
        print("OK - Graph initialized successfully!\n")
    except Exception as e:
        print(f"ERROR - Failed to build graph: {e}")
        return
    
    print("Type 'help' for commands, 'exit' to quit.\n")
    
    while True:
        try:
            user_input = input(f"{COLORS['cyan']}You:{COLORS['reset']} ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        cmd = user_input.strip().lower()
        
        if cmd in ["exit", "quit", "q"]:
            print("Goodbye!")
            break
        
        if cmd == "help":
            print("\nCommands: exit, quit, help, clear")
            print("Just type your question to query the system.\n")
            continue
        
        if not user_input.strip():
            continue
        
        print("Processing...")
        
        inputs = {"input_query": user_input}
        
        try:
            result = await app.ainvoke(inputs)
            
            print("\n" + "-" * 40)
            
            intent = result.get("intent")
            if intent:
                print(f"Intent: {intent.type} ({intent.confidence:.0%})")
            
            emotion = result.get("emotion")
            if emotion:
                print(f"Emotion: {emotion.type} ({emotion.confidence:.0%})")
            
            if result.get("needs_clarification") and result.get("clarification_message"):
                print(f"\nClarification Needed:\n{result['clarification_message']}")
            elif result.get("generated_answer"):
                print(f"\nResponse:\n{result['generated_answer']}")
            
            explanations = result.get("explanations", [])
            if explanations:
                print("\nExplanation:")
                for exp in explanations:
                    print(f"  • Reasoning: {exp.reasoning_type}, Confidence: {exp.confidence:.0%}")
            
            print("-" * 40 + "\n")
            
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except ImportError:
        print("[Note: Install 'rich' for better formatting: pip install rich]")
        asyncio.run(simple_main())
    except KeyboardInterrupt:
        print("\nExiting...")
