#!/usr/bin/env python3
"""
Přímý test MCP serveru v STDIO režimu
"""

import json
import subprocess
import time
import sys
import os

TEST_DIR = r"C:\gz_projekt\data-for-testing\01"

def test_mcp_server_direct():
    """Test MCP serveru přímo přes STDIO"""
    print("🧪 Spouštím MCP server v STDIO režimu...")
    
    start_time = time.time()
    
    try:
        # Spustíme MCP server v STDIO režimu
        process = subprocess.Popen(
            [sys.executable, "preflight_mcp_server.py", "--transport", "stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=os.getcwd()
        )
        
        # Nejprve inicializujeme server
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0"
                }
            }
        }

        print(f"📤 Inicializuji server: {json.dumps(init_request)}")
        process.stdin.write(json.dumps(init_request) + "\n")
        process.stdin.flush()

        # Čekáme na inicializaci
        time.sleep(2)

        # Nejprve získáme seznam nástrojů
        tools_request = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {}
        }

        print(f"📤 Získávám seznam nástrojů: {json.dumps(tools_request)}")
        process.stdin.write(json.dumps(tools_request) + "\n")
        process.stdin.flush()

        time.sleep(1)

        # Pošleme požadavek na spuštění preflight check
        request = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "run_full_preflight_check",
                "arguments": {
                    "source_directory": TEST_DIR
                }
            }
        }

        print(f"📤 Odesílám požadavek: {json.dumps(request)}")

        # Odešleme požadavek
        process.stdin.write(json.dumps(request) + "\n")
        process.stdin.flush()
        
        # Čekáme na odpověď (max 5 minut)
        try:
            stdout, stderr = process.communicate(timeout=300)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"📥 STDOUT: {stdout}")
            if stderr:
                print(f"📥 STDERR: {stderr}")
            
            # Parsujeme odpověď - může být více JSON objektů
            try:
                lines = stdout.strip().split('\n')
                responses = []
                for line in lines:
                    if line.strip():
                        try:
                            response = json.loads(line.strip())
                            responses.append(response)
                        except json.JSONDecodeError:
                            # Možná je více JSON objektů na jednom řádku
                            import re
                            json_objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', line)
                            for json_str in json_objects:
                                try:
                                    response = json.loads(json_str)
                                    responses.append(response)
                                except json.JSONDecodeError:
                                    pass

                print(f"📥 Odpovědi: {responses}")

                # Hledáme odpověď na naši tools/call request (id=3)
                for response in responses:
                    if response.get('id') == 3:
                        if "result" in response:
                            print(f"✅ MCP server dokončen za: {duration:.2f} sekund")
                            print(f"📊 Výsledek: {response['result']}")
                            return duration, response['result']
                        else:
                            print(f"❌ Chyba v odpovědi: {response}")
                            return None, None

                print("❌ Nenalezena odpověď na tools/call request")
                return None, None

            except Exception as e:
                print(f"❌ Chyba při parsování: {e}")
                print(f"📥 Raw output: {stdout}")
                return None, None
                
        except subprocess.TimeoutExpired:
            print("❌ Timeout - server neodpověděl do 5 minut")
            process.kill()
            return None, None
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f"❌ Chyba po {duration:.2f} sekundách: {e}")
        return None, None

if __name__ == "__main__":
    print("🚀 Přímý test MCP serveru")
    print(f"📁 Test adresář: {TEST_DIR}")
    print("=" * 60)
    
    # Test MCP serveru
    test_mcp_server_direct()
    
    print("=" * 60)
    print("✅ Test dokončen!")
