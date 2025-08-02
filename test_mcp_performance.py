#!/usr/bin/env python3
"""
Test výkonu MCP serveru vs. původní verze
"""
import time
import requests
import json
from pathlib import Path

# Test adresář
TEST_DIR = r"C:\gz_projekt\data-for-testing\01"

def test_mcp_server():
    """Test MCP serveru přes HTTP API"""
    print("🧪 Testuji MCP server...")
    
    start_time = time.time()
    
    # Připravíme požadavek pro MCP server (FastMCP používá jiný formát)
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "run_full_preflight_check",
            "arguments": {
                "source_directory": TEST_DIR
            }
        }
    }

    try:
        # Zkusíme různé endpointy
        endpoints = [
            ("POST", "http://127.0.0.1:8050/sse"),
            ("POST", "http://127.0.0.1:8050/message"),
            ("POST", "http://127.0.0.1:8050/"),
            ("GET", "http://127.0.0.1:8050/sse")
        ]

        for method, url in endpoints:
            print(f"🔍 Zkouším {method} {url}")
            try:
                if method == "POST":
                    response = requests.post(
                        url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=30
                    )
                else:
                    response = requests.get(url, timeout=30)

                print(f"📊 Status: {response.status_code}")
                if response.status_code == 200:
                    print(f"✅ Úspěch! Používám {method} {url}")
                    break
                else:
                    print(f"❌ {response.status_code}: {response.text[:100]}")
            except Exception as e:
                print(f"❌ Chyba: {e}")
        else:
            print("❌ Žádný endpoint nefunguje")
            return None, None
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️ MCP server dokončen za: {duration:.2f} sekund")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Status: {response.status_code}")
            print(f"📊 Výsledek: {result.get('result', {}).get('status', 'unknown')}")
        else:
            print(f"❌ Chyba: {response.status_code}")
            print(f"📝 Odpověď: {response.text}")
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ Chyba po {duration:.2f} sekundách: {e}")

def test_original_app():
    """Test původní aplikace"""
    print("🧪 Testuji původní aplikaci...")
    
    # Import původní verze
    import sys
    sys.path.append(str(Path(__file__).parent / "src"))
    
    from src.vinyl_preflight_app_2 import PreflightProcessor
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    API_KEY = os.getenv("OPENROUTER_API_KEY")
    
    def dummy_progress(value, maximum):
        pass
    
    def dummy_status(text):
        print(f"📝 {text}")
    
    start_time = time.time()
    
    try:
        processor = PreflightProcessor(API_KEY, dummy_progress, dummy_status)
        result = processor.run(TEST_DIR)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"⏱️ Původní aplikace dokončena za: {duration:.2f} sekund")
        print(f"📊 Výsledek: {result}")
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ Chyba po {duration:.2f} sekundách: {e}")

if __name__ == "__main__":
    print("🚀 Test MCP serveru")
    print(f"📁 Test adresář: {TEST_DIR}")
    print("=" * 60)

    # Test pouze MCP serveru
    test_mcp_server()

    print("=" * 60)
    print("✅ Test dokončen!")
