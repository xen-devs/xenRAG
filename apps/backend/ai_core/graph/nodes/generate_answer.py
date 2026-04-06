"""
GenerateAnswer Node: Generates the final answer using retrieved context.
Adjusts tone based on detected user emotion.
"""

from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from ai_core.graph.state import GraphState, ReasoningRecord
from ai_core.llm.langchain_wrapper import get_managed_llm


async def generate_answer_node(state: GraphState) -> Dict[str, Any]:
    """
    Generates the final answer from retrieved context.
    Adjusts tone based on the detected user emotion.
    """
    print("--- GENERATE ANSWER NODE ---")
    
    query = state.input_query
    context = state.retrieval_context
    emotion = state.emotion
    product_context = ""
    if state.product_name or state.product_description:
        product_context = "\n\n**Product context:**"
        if state.product_name:
            product_context += f"\n- Product: {state.product_name}"
        if state.product_description:
            product_context += f"\n- Description: {state.product_description}"
    if context and context.merged_results:
        docs_text = "\n\n".join([
            f"[Source: {item.source}] {item.content}" 
            for item in context.merged_results
        ])
    else:
        docs_text = "No context available."
    
    tone_name = "neutral"
    tone_instruction = "Use a neutral, professional tone."
    if emotion:
        if emotion.type == "frustrated":
            tone_name = "empathetic"
            tone_instruction = "Use an empathetic and understanding tone. Acknowledge the user's frustration."
        elif emotion.type == "happy":
            tone_name = "enthusiastic"
            tone_instruction = "Use an enthusiastic and positive tone."
        elif emotion.type == "confused":
            tone_name = "clear"
            tone_instruction = "Use a clear, simple, and helpful tone. Avoid jargon."
    
    # Use managed LLM with failover
    llm = get_managed_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a friendly and knowledgeable assistant that helps product owners and teams understand their customer feedback. The user is NOT an end customer — they are the owner/team behind the product who uploaded customer reviews, feedback, and data to analyze.

**How to behave:**
- Be conversational and natural. Greet users warmly when they greet you.
- If the user says "hi", "hello", "hey" or similar, respond with a friendly greeting and briefly mention you can help them understand their customer feedback (e.g., "Hey! How can I help you today?").
- If the user asks what you can do or your capabilities, explain that you can help them:
  - Analyze customer reviews and feedback
  - Identify common complaints, praises, and trends
  - Summarize what customers are saying about specific features
  - Compare sentiment across different aspects of their product
  - Surface actionable insights from customer data
- For questions, use the provided context to give accurate, well-structured answers.
- If the context has relevant information, use it. If the context is empty or not relevant, say you don't have enough information on that topic and suggest they try rephrasing.
- NEVER mention internal system details like "knowledge base", "context", "documents retrieved", "vector search", "no context available", etc. to the user. Just answer naturally.

**Formatting rules:**
- Use markdown formatting for better readability when appropriate (bold, bullet points, numbered lists, tables).
- Keep tables to a maximum of 4 columns. Keep table cell content short and simple — NO bullet points, NO `<br>` tags, NO HTML inside table cells. If you need to show detailed points, use bullet lists outside the table instead.
- Be concise but thorough. Don't pad your answer unnecessarily, but don't cut important details either.
- Never wrap your response in JSON or code fences unless the user asks for code.
- Never use HTML tags like `<br>`, `<p>`, `<div>` etc. Use only standard markdown.
- Cite or reference specific parts of the context when relevant (e.g., "According to the reviews..." or "Based on the data...").

{tone_instruction}

**Important:** Be helpful, be human, be clear. Your goal is to make the user feel like they're talking to a smart, helpful colleague - not a rigid robot.
{product_context}"""),
        ("user", "Question: {query}\n\nContext from knowledge base:\n{context}")
    ])
    
    chain = prompt | llm
    
    try:
        answer = ""
        async for chunk in chain.astream({
            "query": query,
            "context": docs_text,
            "tone_instruction": tone_instruction,
            "product_context": product_context,
        }):
            answer += chunk.content if hasattr(chunk, "content") else str(chunk)

        answer = answer.strip()
        
        if answer.startswith('```') and answer.endswith('```'):
            answer = answer[3:-3].strip()
        if answer.startswith('"') and answer.endswith('"'):
            answer = answer[1:-1]
        
        print(f"Generated answer with {tone_name} tone ({len(answer)} chars)")
        
        return {
            "generated_answer": answer,
            "private_reasoning": [
                ReasoningRecord(
                    step="GenerateAnswer",
                    summary=f"Generated response using {tone_name} tone based on {emotion.type if emotion else 'neutral'} emotion.",
                    confidence=1.0
                )
            ]
        }
        
    except Exception as e:
        print(f"GenerateAnswer Error: {e}")
        import traceback
        traceback.print_exc()
        
        try:
            simple_prompt = f"Based on this context from the knowledge base:\n{docs_text}\n\nAnswer this question helpfully and conversationally: {query}"
            response = await llm.ainvoke(simple_prompt)
            return {
                "generated_answer": response.content.strip(),
                "private_reasoning": [
                    ReasoningRecord(
                        step="GenerateAnswer",
                        summary="Used fallback simple generation.",
                        confidence=0.7
                    )
                ]
            }
        except Exception as e2:
            return {
                "generated_answer": f"Based on the available customer reviews, I found some relevant information but encountered an issue formatting the response. The context contains {len(context.merged_results) if context and context.merged_results else 0} relevant documents.",
                "private_reasoning": [
                    ReasoningRecord(
                        step="GenerateAnswer",
                        summary=f"Error during generation: {e}. Fallback also failed: {e2}",
                        confidence=0.0
                    )
                ]
            }
