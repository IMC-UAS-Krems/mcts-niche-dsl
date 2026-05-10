#!/usr/bin/env python3
"""
Example script demonstrating Ollama API authentication and usage.

To use with Ollama Cloud (which requires authentication):
1. Set your API key as an environment variable:
   export OLLAMA_API_KEY="your-api-key-here"
   
2. Update OLLAMA_URL if using a cloud endpoint:
   export OLLAMA_URL="https://api.ollama.example.com"

3. Run this script:
   python3 ollama_example.py
"""

import os
from mcts_generator import generate_code, OllamaClient

def test_ollama_connection():
    """Test connection to Ollama with authentication."""
    # Get configuration from environment
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    api_key = os.getenv("OLLAMA_API_KEY")
    ollama_model = os.getenv("OLLAMA_MODEL", "mistral")
    
    print(f"Testing Ollama connection to: {ollama_url}")
    print(f"Model: {ollama_model}")
    print(f"Auth: {'Yes (API key provided)' if api_key else 'No (unauthenticated)'}")
    print()
    
    try:
        client = OllamaClient(
            base_url=ollama_url,
            model=ollama_model,
            api_key=api_key
        )
        
        print("Sending test prompt...")
        response = client.generate("Say 'Hello from MiniZinc MCTS!' and nothing else.", stream=False)
        
        if response:
            print("✓ Connection successful!")
            print(f"Response: {response[:100]}...")
        else:
            print("✗ No response received")
            
    except Exception as e:
        print(f"✗ Connection failed: {e}")

def generate_with_auth():
    """Generate MiniZinc code with optional LLM authentication."""
    nl = "Declare a variable x from 1 to 5, constrain it to be odd, and maximize it."
    
    # Get configuration from environment
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    api_key = os.getenv("OLLAMA_API_KEY")
    ollama_model = os.getenv("OLLAMA_MODEL", "mistral")
    
    print("Generating MiniZinc code with LLM guidance...")
    print(f"NL Prompt: {nl}")
    print()
    
    code = generate_code(
        nl,
        iterations=20,
        use_llm=bool(api_key or os.getenv("OLLAMA_URL")),
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        api_key=api_key
    )
    
    print("Generated code:")
    print(code)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_ollama_connection()
    else:
        generate_with_auth()
