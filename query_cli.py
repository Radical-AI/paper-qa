#!/usr/bin/env python3
"""
Simple query CLI for PaperQA that uses Claude Sonnet and provides timing information.
"""

import argparse
import asyncio
import time
import os
from pathlib import Path
from typing import Dict, Any

from paperqa import Settings, ask
from paperqa.settings import AgentSettings, IndexSettings, ParsingSettings
from paperqa.agents.models import SimpleProfiler


def setup_gpt4o_settings(paper_dir: str) -> Settings:
    """Setup Settings object configured for GPT-4o using SAME settings as profile_indexing.py to avoid reindexing."""
    
    # Use GPT-4o
    gpt_model = "gpt-4o"  # OpenAI GPT-4o
    
    # Use IDENTICAL settings from profile_indexing.py to avoid reindexing
    settings = Settings(
        # Main LLM configuration for GPT-4o
        llm=gpt_model,
        summary_llm=gpt_model,
        
        # Use the same temperature as profile_indexing.py
        temperature=0.5,
        
        # Agent configuration - SAME as profile_indexing.py
        agent=AgentSettings(
            agent_llm=gpt_model,
            index=IndexSettings(
                paper_directory=paper_dir,
                index_directory=f"{paper_dir}/index",
                manifest_file=f"{paper_dir}/manifest.csv",
            )
        ),
        verbosity=1,  # Some logging but not too verbose
    )
    
    return settings


def format_timing_info(response) -> str:
    """Format timing and profiling information from the response."""
    output = []
    
    # Basic timing
    if hasattr(response, 'duration') and response.duration:
        output.append(f"Total Duration: {response.duration:.2f}s")
    
    # Detailed timing info if available
    if hasattr(response, 'timing_info') and response.timing_info:
        output.append("\nDetailed Timing:")
        for operation, timings in response.timing_info.items():
            if isinstance(timings, dict):
                for metric, value in timings.items():
                    output.append(f"  {operation} ({metric}): {value:.3f}s")
            else:
                output.append(f"  {operation}: {timings:.3f}s")
    
    # Status and stats
    if hasattr(response, 'status'):
        output.append(f"\nAgent Status: {response.status}")
    
    if hasattr(response, 'stats') and response.stats:
        output.append("\nStats:")
        for key, value in response.stats.items():
            output.append(f"  {key}: {value}")
    
    # Session info
    if hasattr(response, 'session'):
        session = response.session
        if hasattr(session, 'contexts') and session.contexts:
            output.append(f"\nSources Used: {len(session.contexts)}")
        
        if hasattr(session, 'tool_history') and session.tool_history:
            tool_calls = sum(len(tools) for tools in session.tool_history)
            output.append(f"Tool Calls Made: {tool_calls}")
    
    return "\n".join(output) if output else "No timing information available"


async def main():
    parser = argparse.ArgumentParser(
        description="Query papers using PaperQA with GPT-4o and timing info"
    )
    parser.add_argument(
        "query", 
        help="Question to ask about the papers"
    )
    parser.add_argument(
        "--paper-dir", 
        default="/data/kai/radical-scratch/literature-warehouse/openalex",
        help="Directory containing papers (default: from profile_indexing.py)"
    )
    parser.add_argument(
        "--no-timing", 
        action="store_true",
        help="Skip timing information output"
    )
    
    args = parser.parse_args()
    
    # Check if paper directory exists
    paper_path = Path(args.paper_dir)
    if not paper_path.exists():
        print(f"Error: Paper directory '{args.paper_dir}' does not exist")
        return 1
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it with: export OPENAI_API_KEY=your_key_here")
        return 1
    
    print(f"📚 Querying papers in: {args.paper_dir}")
    print(f"❓ Question: {args.query}")
    print("🤖 Using GPT-4o...")
    print("=" * 60)
    
    # Setup settings
    settings = setup_gpt4o_settings(args.paper_dir)
    
    # Record start time
    start_time = time.time()
    
    try:
        # Run the query - ask() returns either a result or a Task depending on event loop
        response = ask(args.query, settings)
        
        # If it returns a Task (when event loop is running), await it
        if isinstance(response, asyncio.Task):
            response = await response
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Display the answer
        print("\n📋 ANSWER:")
        print("-" * 40)
        if hasattr(response, 'session') and hasattr(response.session, 'answer'):
            print(response.session.answer)
        else:
            print(str(response))
        
        if not args.no_timing:
            print("\n⏱️  TIMING & PROFILING:")
            print("-" * 40)
            print(f"Wall Clock Time: {total_time:.2f}s")
            print(format_timing_info(response))
        
        # Show citations if available
        if hasattr(response, 'session') and hasattr(response.session, 'contexts'):
            contexts = response.session.contexts
            if contexts:
                print(f"\n📖 SOURCES ({len(contexts)}):")
                print("-" * 40)
                for i, context in enumerate(contexts, 1):
                    if hasattr(context, 'text') and hasattr(context.text, 'name'):
                        print(f"{i}. {context.text.name}")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
