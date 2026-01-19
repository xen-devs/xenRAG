#!/usr/bin/env python3
"""
Quick test to verify the clarification follow-up fix works correctly.
"""

import asyncio
from datetime import datetime
from xenrag.graph.state import GraphState, ConversationMessage
from xenrag.guardrails.topic_rail import validate_topic


def test_topic_validation():
    """Test that topic validation correctly handles clarification follow-ups."""
    
    print("=" * 70)
    print("Testing Clarification Follow-up Fix")
    print("=" * 70)
    print()
    
    # Test 1: Normal query without clarification
    print("Test 1: Normal query (no clarification context)")
    print("-" * 70)
    query1 = "yeah I want the Most Negative lang one"
    result1 = validate_topic(query1, pending_clarification=False, conversation_history=[])
    print(f"Query: {query1}")
    print(f"Pending Clarification: False")
    print(f"Result: {'✅ PASSED' if result1.is_on_topic else '❌ BLOCKED'}")
    print(f"Confidence: {result1.topic_confidence:.2f}")
    if not result1.is_on_topic:
        print(f"Reason: {result1.redirect_message}")
    print()
    
    # Test 2: Same query WITH clarification context
    print("Test 2: Same query WITH clarification context")
    print("-" * 70)
    query2 = "yeah I want the Most Negative lang one"
    conversation_history = [
        ConversationMessage(
            role="user",
            content="What is the most bad review about the product?",
            timestamp=datetime.now().isoformat()
        ),
        ConversationMessage(
            role="clarification",
            content="I need more details. Do you want the lowest rating or most negative language?",
            timestamp=datetime.now().isoformat()
        )
    ]
    result2 = validate_topic(
        query2, 
        pending_clarification=True, 
        conversation_history=conversation_history
    )
    print(f"Query: {query2}")
    print(f"Pending Clarification: True")
    print(f"Conversation History: {len(conversation_history)} messages")
    print(f"Result: {'✅ PASSED' if result2.is_on_topic else '❌ BLOCKED'}")
    print(f"Confidence: {result2.topic_confidence:.2f}")
    if not result2.is_on_topic:
        print(f"Reason: {result2.redirect_message}")
    print()
    
    # Test 3: Genuine off-topic query (even with clarification)
    print("Test 3: Genuine off-topic query (with clarification context)")
    print("-" * 70)
    query3 = "What's the weather like today?"
    result3 = validate_topic(
        query3,
        pending_clarification=True,
        conversation_history=conversation_history
    )
    print(f"Query: {query3}")
    print(f"Pending Clarification: True")
    print(f"Result: {'✅ PASSED' if result3.is_on_topic else '❌ BLOCKED (Expected)'}")
    print(f"Confidence: {result3.topic_confidence:.2f}")
    if not result3.is_on_topic:
        print(f"Reason: {result3.redirect_message}")
    print()
    
    # Test 4: Various follow-up phrases
    print("Test 4: Various follow-up phrases (with clarification)")
    print("-" * 70)
    follow_ups = [
        "yes the negative one",
        "no I want the positive",
        "the worst review",
        "the one with lowest rating",
        "yeah that one"
    ]
    
    for follow_up in follow_ups:
        result = validate_topic(
            follow_up,
            pending_clarification=True,
            conversation_history=conversation_history
        )
        status = "✅" if result.is_on_topic else "❌"
        print(f"{status} '{follow_up}' -> {result.topic_confidence:.2f}")
    
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print("✅ Test 2 should PASS (clarification follow-up)")
    print("❌ Test 1 should BLOCK (no context)")
    print("❌ Test 3 should BLOCK (genuinely off-topic)")
    print("✅ Test 4 should mostly PASS (valid follow-ups)")
    print()


if __name__ == "__main__":
    test_topic_validation()
