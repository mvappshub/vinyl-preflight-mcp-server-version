# --------------------------------------------------------------------------
# preflight_mcp_server.py
# --------------------------------------------------------------------------
"""
Vinyl Preflight Processor MCP Server v2.5 - Finální Produkční Verze

- MCP server pro kompletní preflight validaci vinylových projektů
- Implementuje plnou podporu pro "Consolidated Side" mód (jeden WAV na stranu)
- Automaticky detekuje mód projektu a volí správnou validační strategii
- Poskytuje přesné a relevantní reporty pro všechny typy projektů
"""
import asyncio
import os
import sys
import json
import time
import logging
import tempfile
import traceback
import csv
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Callable
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf

# Import matplotlib pouze pokud je k dispozici
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logging.warning("Matplotlib není nainstalován. Vizualizace nebudou dostupné.")

from mcp.server.fastmcp import FastMCP, Context
from dotenv import load_dotenv

# Globální statistiky
class Stats:
    def __init__(self):
        self.analysis_count = 0
        self.total_projects_processed = 0
        self.total_processing_time = 0.0
        self.last_report_path = None

# Globální instance statistik
global_stats = Stats()

# Importujeme původní moduly a třídy
try:
    from src.vinyl_preflight_app_2 import (
        PreflightProcessor as OriginalPreflightProcessor,
        _get_wav_duration_worker,
        seconds_to_mmss,
        CSV_HEADERS
    )
except ImportError:
    logging.error("Nelze importovat vinyl_preflight_app. Ujistěte se, že soubor existuje ve stejném adresáři.")
    sys.exit(1)

# Import pro optimalizovanou verzi
import concurrent.futures

# ------------------------------------------------------------
# Optimalizovaná verze PreflightProcessor pro MCP server
# ------------------------------------------------------------
class PreflightProcessor(OriginalPreflightProcessor):
    """
    Optimalizovaná verze PreflightProcessor pro MCP server.
    Nahrazuje multiprocessing za ThreadPoolExecutor pro lepší kompatibilitu s asyncio.
    """

    def _get_all_wav_durations(self, projects: dict):
        """
        Optimalizovaná verze pro MCP server - používá ThreadPoolExecutor místo multiprocessing.
        """
        all_wav_paths = [wav for proj in projects.values() for wav in proj['wavs']]
        durations = {}

        # Použijeme ThreadPoolExecutor místo multiprocessing.Pool()
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(all_wav_paths))) as executor:
            # Odešleme všechny úlohy
            future_to_path = {executor.submit(_get_wav_duration_worker, wav_path): wav_path for wav_path in all_wav_paths}

            # Sbíráme výsledky
            for future in concurrent.futures.as_completed(future_to_path):
                try:
                    path_str, duration = future.result()
                    durations[path_str] = duration
                except Exception as e:
                    wav_path = future_to_path[future]
                    logger.error(f"Chyba při zpracování WAV souboru {wav_path}: {e}")
                    durations[wav_path.as_posix()] = None

        return durations

# ------------------------------------------------------------
# Konfigurace
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vinyl-preflight-mcp")

# Diagnostika - vypiš informace o prostředí
logger.info(f"Aktuální pracovní adresář: {os.getcwd()}")
logger.info(f"Umístění skriptu: {Path(__file__).resolve()}")
logger.info(f"Obsah aktuálního adresáře: {os.listdir('.')}")

# Načtení konfigurace
# Zkusíme několik možných umístění .env souboru
possible_paths = [
    Path('.env'),  # Aktuální adresář
    Path(__file__).parent / '.env',  # Stejný adresář jako skript
    Path(__file__).resolve().parent / '.env',  # Absolutní cesta ke stejnému adresáři
    Path.home() / '.env',  # Domovský adresář uživatele
]

env_loaded = False
for path in possible_paths:
    logger.info(f"Kontroluji cestu pro .env: {path}")
    if path.exists():
        logger.info(f"Nalezen .env soubor na cestě: {path}")
        load_dotenv(dotenv_path=path)
        env_loaded = True
        break

if not env_loaded:
    logger.error("Soubor .env nebyl nalezen na žádné z očekávaných cest!")
    raise ValueError("Soubor .env nebyl nalezen. Ujistěte se, že existuje v jednom z očekávaných umístění.")

API_KEY = os.getenv("OPENROUTER_API_KEY")
logger.info(f"Načtený API klíč: {'***' if API_KEY else 'None'}")

if not API_KEY:
    logger.error("Chybí OPENROUTER_API_KEY v .env souboru!")
    raise ValueError("API klíč (OPENROUTER_API_KEY) nebyl nalezen v .env souboru.")

# Vytvoření MCP serveru
mcp = FastMCP(
    "vinyl-preflight-processor",
    description="Komplexní preflight validace pro vinylové projekty, včetně extrakce z PDF a porovnání s WAV.",
    version="2.5.0"
)

# ------------------------------------------------------------
# Třída pro vizualizaci výsledků (pouze pokud je matplotlib dostupný)
# ------------------------------------------------------------
class ResultsVisualizer:
    """Generuje vizualizace výsledků pro lepší přehlednost"""
    
    @staticmethod
    def create_validation_chart(results: List[Dict], output_path: Path) -> bool:
        """Vytvoří sloupcový graf s porovnáním délek PDF vs WAV
        
        Vrací True pokud se podařilo vytvořit graf, jinak False
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib není dostupný, vizualizace nelze vytvořit")
            return False
        
        # Příprava dat
        items = []
        pdf_durations = []
        wav_durations = []
        differences = []
        
        for row in results:
            if row.get('pdf_duration_sec') is not None and row.get('wav_duration_sec') is not None:
                items.append(row.get('validation_item', 'Neznámá položka')[:20])  # Omezení délky pro zobrazení
                pdf_durations.append(row.get('pdf_duration_sec', 0))
                wav_durations.append(row.get('wav_duration_sec', 0))
                differences.append(row.get('difference_sec', 0))
        
        if not items:
            logger.warning("Žádná platná data pro vizualizaci")
            return False
        
        try:
            # Vytvoření grafu
            fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
            
            x = np.arange(len(items))
            width = 0.35
            
            # Sloupce pro PDF a WAV
            rects1 = ax.bar(x - width/2, pdf_durations, width, label='PDF Délka', color='#3498db')
            rects2 = ax.bar(x + width/2, wav_durations, width, label='WAV Délka', color='#2ecc71')
            
            # Přidání rozdílů jako textu
            for i, (rect1, rect2, diff) in enumerate(zip(rects1, rects2, differences)):
                height = max(rect1.get_height(), rect2.get_height())
                ax.annotate(f'{diff:.1f}s', 
                            xy=(rect2.get_x() + rect2.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom',
                            color='red' if abs(diff) > 10 else 'green',
                            fontsize=8)
            
            # Nastavení grafu
            ax.set_ylabel('Délka (sekundy)')
            ax.set_title('Porovnání délek PDF vs WAV')
            ax.set_xticks(x)
            ax.set_xticklabels(items, rotation=45, ha='right')
            ax.legend()
            
            fig.tight_layout()
            plt.savefig(output_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"Vizualizace uložena do: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Chyba při vytváření vizualizace: {e}")
            return False

# ------------------------------------------------------------
# Životní cyklus serveru
# ------------------------------------------------------------
@asynccontextmanager
async def preflight_lifespan(server: FastMCP):
    logger.info("⚡ Vinyl Preflight Processor MCP Server starting...")

    # Vytvoříme jednoduchý objekt pro statistiky
    class Stats:
        def __init__(self):
            self.analysis_count = 0
            self.total_projects_processed = 0
            self.total_processing_time = 0.0
            self.last_report_path = None

    stats = Stats()

    try:
        yield stats
    finally:
        logger.info(f"📊 Celkové statistiky: {stats.analysis_count} analýz, {stats.total_projects_processed} projektů zpracováno")

# ------------------------------------------------------------
# Hlavní MCP nástroj
# ------------------------------------------------------------
@mcp.tool()
async def run_full_preflight_check(
    source_directory: str,
    ctx: Context
) -> Dict[str, Any]:
    """
    Spustí kompletní preflight validaci pro projekt v daném adresáři.
    Proskenuje PDF a WAV soubory, extrahuje data pomocí AI, porovná délky
    a vygeneruje finální CSV report.

    Args:
        source_directory: Absolutní cesta k adresáři obsahujícímu projekt (nebo archivy).

    Returns:
        Slovník s výsledky validace včetně cesty k reportu a vizualizacím.
    """

    start_time = time.time()

    # Přístup ke statistikám
    stats = global_stats

    result = {
        "status": "success",
        "source_directory": source_directory,
        "report_path": "",
        "visualization_path": "",
        "project_count": 0,
        "validation_results": [],
        "processing_time": 0.0,
        "message": ""
    }

    try:
        # Validace vstupu
        if not os.path.exists(source_directory):
            raise FileNotFoundError(f"Adresář nebyl nalezen: {source_directory}")

        if not os.path.isdir(source_directory):
            raise ValueError(f"Zadaná cesta není adresář: {source_directory}")

        await ctx.info(f"🔍 Zahajuji preflight analýzu adresáře: {source_directory}")

        # Vytvoření instance procesoru s optimalizovanými callbacky
        progress_updates = []
        status_messages = []

        # Získáme referenci na event loop pro použití v callbackech
        loop = asyncio.get_event_loop()

        def progress_callback(value: int, maximum: int):
            progress_updates.append((value, maximum))
            # Pouze logujeme do konzole pro rychlost - MCP zprávy jsou pomalé
            progress_pct = (value / maximum) * 100 if maximum > 0 else 0
            logger.info(f"🔄 Průběh: {progress_pct:.1f}% ({value}/{maximum})")

        def status_callback(text: str):
            status_messages.append(text)
            # Pouze logujeme do konzole pro rychlost - MCP zprávy jsou pomalé
            logger.info(f"ℹ️ {text}")

        processor = PreflightProcessor(API_KEY, progress_callback, status_callback)

        # Debug informace
        await ctx.info(f"🔧 Používám optimalizovanou verzi s ThreadPoolExecutor")
        await ctx.info(f"📁 Zpracovávám adresář: {source_directory}")

        # Spustíme synchronní operaci v thread poolu, aby neblokovala event loop
        try:
            report_path = await loop.run_in_executor(None, processor.run, source_directory)
        except Exception as e:
            logger.error(f"Chyba při spuštění procesoru: {e}")
            await ctx.error(f"Chyba při zpracování: {str(e)}")
            return {
                "status": "error",
                "message": f"Chyba při zpracování: {str(e)}",
                "source_directory": source_directory
            }
        
        if report_path:
            # Načtení výsledků z CSV reportu
            validation_results = []
            try:
                with open(report_path, 'r', newline='', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        validation_results.append(row)
            except Exception as e:
                logger.error(f"Chyba při čtení CSV souboru: {e}")
                result["status"] = "error"
                result["message"] = f"Chyba při čtení reportu: {str(e)}"
                return result
            
            # Vytvoření vizualizace (pokud je matplotlib dostupný)
            visualization_path = ""
            if MATPLOTLIB_AVAILABLE:
                temp_dir = Path(tempfile.gettempdir()) / "vinyl_preflight"
                temp_dir.mkdir(exist_ok=True)
                
                timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
                viz_path = temp_dir / f"preflight_viz_{timestamp}.png"
                
                if ResultsVisualizer.create_validation_chart(validation_results, viz_path):
                    # Lokální URL pro vizualizaci
                    visualization_path = f"file:///{viz_path}".replace('\\', '/')
                    await ctx.info(f"📈 Vizualizace dostupná na: {visualization_path}")
            
            # Aktualizace statistik
            stats.analysis_count += 1
            stats.total_projects_processed += len(set(r.get('project_title') for r in validation_results))
            stats.last_report_path = report_path
            
            # Příprava výsledků
            project_count = len(set(r.get('project_title') for r in validation_results))
            
            result.update({
                "report_path": report_path,
                "visualization_path": visualization_path,
                "project_count": project_count,
                "validation_results": validation_results,
                "message": f"Analýza dokončena. Zpracováno {project_count} projektů. Report uložen do: {report_path}"
            })
            
            await ctx.info(f"✅ Preflight analýza dokončena. Zpracováno {project_count} projektů.")
            await ctx.info(f"📊 Report uložen do: {report_path}")
            await ctx.info(f"⚡ Optimalizovaná verze pro MCP server (ThreadPoolExecutor místo multiprocessing)")
            
        else:
            result["status"] = "error"
            result["message"] = "Během zpracování nastala chyba a report nebyl vygenerován."
            await ctx.error("❌ Během zpracování nastala chyba a report nebyl vygenerován.")
            
    except Exception as e:
        logger.error(f"Kritická chyba v nástroji run_full_preflight_check: {e}", exc_info=True)
        result["status"] = "error"
        result["message"] = f"Kritická chyba: {str(e)}"
        await ctx.error(f"❌ Nastala neočekávaná chyba: {str(e)}")
    
    finally:
        result["processing_time"] = round(time.time() - start_time, 2)

        # Bezpečná aktualizace statistik
        try:
            if hasattr(stats, 'total_processing_time'):
                stats.total_processing_time += result["processing_time"]
            else:
                logger.warning("Atribut total_processing_time neexistuje, vytvářím ho")
                stats.total_processing_time = result["processing_time"]
        except Exception as e:
            logger.warning(f"Chyba při aktualizaci statistik: {e}")

    return result

@mcp.tool()
async def get_server_stats(ctx: Context) -> Dict[str, Any]:
    """
    Vrátí statistiky o využití serveru.

    Returns:
        Slovník se statistikami serveru.
    """
    try:
        stats = global_stats

        # Bezpečné získání atributů s výchozími hodnotami
        analysis_count = getattr(stats, 'analysis_count', 0)
        total_projects_processed = getattr(stats, 'total_projects_processed', 0)
        total_processing_time = getattr(stats, 'total_processing_time', 0.0)
        last_report_path = getattr(stats, 'last_report_path', None)

        # Výpočet průměrného času zpracování
        average_time = round(total_processing_time / max(1, analysis_count), 2) if analysis_count > 0 else 0.0

        return {
            "analysis_count": analysis_count,
            "total_projects_processed": total_projects_processed,
            "total_processing_time": round(total_processing_time, 2),
            "last_report_path": last_report_path,
            "average_processing_time": average_time,
            "matplotlib_available": MATPLOTLIB_AVAILABLE
        }
    except Exception as e:
        logger.error(f"Chyba při získávání statistik: {e}")
        return {
            "status": "error",
            "message": f"Chyba při získávání statistik: {str(e)}",
            "analysis_count": 0,
            "total_projects_processed": 0,
            "total_processing_time": 0.0,
            "average_processing_time": 0.0,
            "matplotlib_available": MATPLOTLIB_AVAILABLE
        }

# ------------------------------------------------------------
# Spuštění serveru
# ------------------------------------------------------------
mcp.lifespan = preflight_lifespan

# Alternativní způsob spuštění (pokud výše uvedené nefunguje)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Vinyl Preflight Processor MCP Server v2.5")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--transport", choices=["stdio", "http"], default="http")
    
    args = parser.parse_args()
    
    if args.transport == "http":
        logger.info(f"🚀 Spouštím MCP server v HTTP režimu na {args.host}:{args.port}")
        # Použijeme uvicorn pro spuštění HTTP serveru s SSE
        import uvicorn
        app = mcp.sse_app()
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        logger.info("🚀 Spouštím MCP server v STDIO režimu")
        mcp.run(transport="stdio")