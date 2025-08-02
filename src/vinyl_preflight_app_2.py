#!/usr/bin/env python3
"""
Vinyl Preflight Processor v3.0 - Robustní a optimalizovaná verze
- Vylepšená detekce projektových módů
- Optimalizované zpracování velkých souborů
- Centralizovaná konfigurace
- Robustní API komunikace
- Eliminace duplicitního kódu
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import concurrent.futures
import time
import os
import sys
import json
import csv
from pathlib import Path
import math
from typing import Callable, Dict, List, Tuple, Optional, Any
import shutil
import tempfile
import zipfile
import re
import logging
from dataclasses import dataclass
from enum import Enum, auto

import requests
import soundfile as sf
from dotenv import load_dotenv
import multiprocessing as mp
import fitz
from thefuzz import fuzz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Konfigurace
@dataclass
class Config:
    """Centralizovaná konfigurace aplikace"""
    MODEL_NAME: str = "google/gemini-2.5-flash"
    API_URL: str = "https://openrouter.ai/api/v1/chat/completions"
    MAX_PDFS_PER_BATCH: int = 50
    MAX_PARALLEL_API_REQUESTS: int = 10
    API_REQUEST_TIMEOUT: int = 180
    VALIDATION_TOLERANCE_SECONDS: float = 10.0
    SIMILARITY_THRESHOLD: int = 70
    MAX_RETRIES: int = 3
    RETRY_DELAY: float = 1.0
    MAX_BATCH_SIZE_MB: int = 10  # Max velikost dávky v MB
    CSV_HEADERS: List[str] = None
    
    def __post_init__(self):
        if self.CSV_HEADERS is None:
            self.CSV_HEADERS = [
                "project_title", "status", "validation_item", "item_type",
                "pdf_duration_mmss", "wav_duration_mmss", "difference_mmss",
                "pdf_duration_sec", "wav_duration_sec", "difference_sec",
                "pdf_source", "wav_source", "notes"
            ]

class ProjectMode(Enum):
    """Typy projektových módů"""
    CONSOLIDATED = auto()
    INDIVIDUAL = auto()
    UNKNOWN = auto()

# Inicializace rarfile s kontrolou dostupnosti
rarfile = None
try:
    import rarfile
    if not shutil.which("unrar"):
        logger.warning("Příkaz 'unrar' nebyl nalezen v systémové PATH. Extrakce RAR nemusí fungovat.")
except ImportError:
    logger.warning("Knihovna 'rarfile' není nainstalována. Podpora pro .rar archivy je vypnuta.")

def seconds_to_mmss(seconds: Optional[float]) -> str:
    """Převede sekundy na formát MM:SS"""
    if seconds is None: return "N/A"
    if not isinstance(seconds, (int, float)): return "Chyba"
    sign = '-' if seconds < 0 else '+'
    seconds = abs(seconds)
    minutes, remaining_seconds = divmod(round(seconds), 60)
    return f"{sign}{minutes:02d}:{remaining_seconds:02d}"

def _get_wav_duration_worker(filepath: Path) -> Tuple[str, Optional[float]]:
    """Získá délku WAV souboru (worker pro multiprocessing)"""
    try:
        info = sf.info(filepath)
        return filepath.as_posix(), info.duration
    except Exception as e:
        logger.error(f"Chyba při čtení WAV '{filepath.name}': {e}")
        return filepath.as_posix(), None

def normalize_string(s: str) -> str:
    """Normalizuje řetězec pro porovnání"""
    s = s.lower()
    s = re.sub(r'^\d+[\s.-]*', '', s)
    s = re.sub(r'[\W_]+', ' ', s)
    return " ".join(s.split())

class PreflightProcessor:
    """Hlavní procesor pro validaci vinylových projektů"""
    
    def __init__(self, api_key: str, config: Config, 
                 progress_callback: Callable, status_callback: Callable):
        if not api_key:
            raise ValueError("API klíč nesmí být prázdný.")
        self.api_key = api_key
        self.config = config
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.headers = {
            "Authorization": f"Bearer {self.api_key}", 
            "Content-Type": "application/json"
        }

    def run(self, source_directory: str) -> Optional[str]:
        """Hlavní metoda pro spuštění zpracování"""
        try:
            start_time = time.time()
            output_dir = Path(__file__).resolve().parent.parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_filename = output_dir / f"Preflight_Report_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"

            with tempfile.TemporaryDirectory(prefix="preflight_") as tmpdir:
                temp_path = Path(tmpdir)
                
                # Fáze 1: Příprava pracovního prostoru
                self.status_callback("1/5 Připravuji pracovní prostor a extrahuji archivy...")
                self._prepare_workspace(Path(source_directory), temp_path)

                # Fáze 2: Skenování a příprava projektů
                self.status_callback("2/5 Skenuji soubory a připravuji projekty...")
                projects = self._scan_and_group_projects(temp_path)
                if not projects:
                    self.status_callback("Nebyly nalezeny žádné relevantní projekty.")
                    return None

                # Fáze 3: Získání délky WAV souborů
                self.status_callback("3/5 Zjišťuji délky WAV souborů...")
                wav_durations = self._get_all_wav_durations(projects)

                # Fáze 4: Zpracování PDF souborů
                self.status_callback("4/5 Vytvářím dávky PDF pro efektivní extrakci...")
                pdf_batches = self._create_pdf_batches(projects)
                
                self.status_callback(f"4/5 Zpracovávám {len(pdf_batches)} dávek PDF...")
                extracted_pdf_data = self._process_all_pdf_batches(pdf_batches)
                
                # Fáze 5: Validace a zápis reportu
                self.status_callback("5/5 Provádím finální validaci a zápis do reportu...")
                self._generate_report(projects, extracted_pdf_data, wav_durations, output_filename)
                
            end_time = time.time()
            self.status_callback(f"Hotovo! Celkový čas: {end_time - start_time:.2f} s. Report uložen do: {output_filename}")
            return str(output_filename)

        except Exception as e:
            logger.exception("Chyba při zpracování:")
            self.status_callback(f"Chyba: Proces byl přerušen. {e}")
            return None

    def _prepare_workspace(self, source_root: Path, temp_root: Path) -> None:
        """Připraví pracovní prostor s optimalizovaným kopírováním souborů"""
        for item in source_root.iterdir():
            if item.is_dir():
                # Kopírujeme pouze relevantní soubory
                self._copy_relevant_files(item, temp_root / item.name)
            elif item.is_file():
                target_dir = temp_root / item.stem
                if item.suffix.lower() == '.zip':
                    logger.info(f"Extrahuji ZIP: {item.name}")
                    target_dir.mkdir(parents=True, exist_ok=True)
                    with zipfile.ZipFile(item, 'r') as zip_ref:
                        zip_ref.extractall(target_dir)
                elif item.suffix.lower() == '.rar' and rarfile:
                    logger.info(f"Extrahuji RAR: {item.name}")
                    target_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        with rarfile.RarFile(item, 'r') as rar_ref:
                            rar_ref.extractall(target_dir)
                    except Exception as e:
                        logger.error(f"CHYBA: Nepodařilo se extrahovat RAR soubor '{item.name}'. Důvod: {e}")

    def _copy_relevant_files(self, src_dir: Path, dst_dir: Path) -> None:
        """Kopíruje pouze relevantní soubory (PDF, WAV) z adresáře"""
        dst_dir.mkdir(parents=True, exist_ok=True)
        for item in src_dir.rglob("*"):
            if item.is_file() and item.suffix.lower() in ('.pdf', '.wav'):
                rel_path = item.relative_to(src_dir)
                dst_path = dst_dir / rel_path
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, dst_path)

    def _scan_and_group_projects(self, root_dir: Path) -> Dict[str, Dict[str, List[Path]]]:
        """Seskupí soubory do projektů"""
        projects = {}
        for item in root_dir.iterdir():
            if item.is_dir():
                pdfs = list(item.rglob("*.pdf"))
                wavs = list(item.rglob("*.wav"))
                if pdfs and wavs:
                    projects[item.name] = {'pdfs': pdfs, 'wavs': wavs}
        return projects

    def _get_all_wav_durations(self, projects: dict) -> Dict[str, Optional[float]]:
        """Získá délky všech WAV souborů"""
        all_wav_paths = [wav for proj in projects.values() for wav in proj['wavs']]
        durations = {}
        with mp.Pool() as pool:
            results = pool.map(_get_wav_duration_worker, all_wav_paths)
        for path_str, duration in results:
            durations[path_str] = duration
        return durations

    def _create_pdf_batches(self, projects: dict) -> List[List[Path]]:
        """Vytvoří dávky PDF s omezením velikosti"""
        all_pdfs = [pdf_path for proj in projects.values() for pdf_path in proj['pdfs']]
        batches = []
        current_batch = []
        current_size = 0
        
        for pdf_path in all_pdfs:
            pdf_size = pdf_path.stat().st_size / (1024 * 1024)  # Size in MB
            
            # Pokud přidání tohoto PDF překročí limit, začni novou dávku
            if current_size + pdf_size > self.config.MAX_BATCH_SIZE_MB and current_batch:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
                
            current_batch.append(pdf_path)
            current_size += pdf_size
            
            # Pokud dávka dosáhla max počtu souborů, začni novou
            if len(current_batch) >= self.config.MAX_PDFS_PER_BATCH:
                batches.append(current_batch)
                current_batch = []
                current_size = 0
        
        if current_batch:
            batches.append(current_batch)
            
        return batches

    def _process_all_pdf_batches(self, batches: list) -> dict:
        """Zpracuje všechny dávky PDF s retry mechanismem"""
        all_results = {}
        total_batches = len(batches)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.MAX_PARALLEL_API_REQUESTS) as executor:
            future_to_batch = {
                executor.submit(self._process_single_extraction_batch_with_retry, batch): i 
                for i, batch in enumerate(batches)
            }
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_batch)):
                self.status_callback(f"4/5 Zpracovávám PDF dávku {i+1}/{total_batches}...")
                self.progress_callback(i + 1, total_batches)
                try:
                    batch_results = future.result()
                    if batch_results:
                        for result in batch_results:
                            all_results[result['source_identifier']] = result
                except Exception as e:
                    logger.error(f"Chyba při zpracování dávky: {e}")
        return all_results

    def _process_single_extraction_batch_with_retry(self, batch: List[Path]) -> Optional[List[Dict]]:
        """Zpracuje jednu dávku PDF s retry mechanismem"""
        last_exception = None
        for attempt in range(self.config.MAX_RETRIES):
            try:
                return self._process_single_extraction_batch(batch)
            except Exception as e:
                last_exception = e
                if attempt < self.config.MAX_RETRIES - 1:
                    delay = self.config.RETRY_DELAY * (2 ** attempt)
                    logger.warning(f"Opakuji pokus ({attempt + 1}/{self.config.MAX_RETRIES}) za {delay}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"Chyba API po {self.config.MAX_RETRIES} pokusech: {e}")
        
        # Pokud všechny pokusy selhaly, vrátíme chybu pro všechny dokumenty v dávce
        return [{
            "source_identifier": pdf_path.as_posix(),
            "status": "error",
            "data": [],
            "error_message": str(last_exception)
        } for pdf_path in batch]

    def _process_single_extraction_batch(self, batch: List[Path]) -> Optional[List[Dict]]:
        """Zpracuje jednu dávku PDF s validací odpovědi"""
        documents_to_process = []
        for pdf_path in batch:
            try:
                with fitz.open(pdf_path) as doc:
                    text = "".join(page.get_text() for page in doc)
                if not text.strip():
                    text = f"VAROVÁNÍ: PDF soubor '{pdf_path.name}' neobsahuje žádný extrahovatelný text."
                
                documents_to_process.append({
                    "identifier": pdf_path.as_posix(), 
                    "content": text
                })
            except Exception as e:
                 logger.error(f"Chyba při čtení PDF pro dávku: {pdf_path}, {e}")
                 documents_to_process.append({
                     "identifier": pdf_path.as_posix(), 
                     "content": f"CHYBA: Nelze přečíst soubor. {e}"
                 })

        if not documents_to_process: 
            return None

        prompt = f"""
Jsi expert na hudební mastering. Tvým úkolem je precizně extrahovat informace o skladbách z několika dokumentů.
Analyzuj KAŽDÝ dokument v poli a vrať VÝHRADNĚ JEDEN JSON objekt s klíčem "results". Hodnota klíče "results" bude pole, kde každý prvek reprezentuje jeden zpracovaný dokument.

Struktura pro každý prvek v poli "results":
- "source_identifier": Unikátní identifikátor dokumentu.
- "status": 'success' nebo 'error'.
- "data": Pokud 'success', zde bude pole skladeb. Každá skladba musí obsahovat "side", "track_number", "title", "duration_seconds".
- "error_message": Popis chyby, pokud status je 'error'.

Zde jsou dokumenty ke zpracování:
---
{json.dumps(documents_to_process, indent=2, ensure_ascii=False)}
---
"""
        payload = {
            "model": self.config.MODEL_NAME, 
            "messages": [{"role": "user", "content": prompt}], 
            "response_format": {"type": "json_object"}, 
            "temperature": 0.0
        }
        
        try:
            response = requests.post(
                self.config.API_URL, 
                headers=self.headers, 
                json=payload, 
                timeout=self.config.API_REQUEST_TIMEOUT
            )
            response.raise_for_status()
            content_str = response.json()["choices"][0]["message"]["content"]
            parsed_response = json.loads(content_str)
            
            # Validace struktury odpovědi
            if not isinstance(parsed_response, dict) or "results" not in parsed_response:
                raise ValueError("Neplatná struktura odpovědi od LLM")
                
            results = parsed_response["results"]
            if not isinstance(results, list):
                raise ValueError("Odpověď neobsahuje pole 'results'")
                
            # Validace každé položky v results
            for item in results:
                if not all(key in item for key in ["source_identifier", "status"]):
                    raise ValueError("Chybí povinná pole v odpovědi")
                if item["status"] == "success" and ("data" not in item or not isinstance(item["data"], list)):
                    raise ValueError("Neplatná struktura dat při úspěšné odpovědi")
                    
            return results
            
        except Exception as e:
            logger.error(f"Chyba API volání pro dávku: {e}")
            return [{
                "source_identifier": d["identifier"], 
                "status": "error", 
                "data": [], 
                "error_message": str(e)
            } for d in documents_to_process]

    def _detect_project_mode(self, project_info: Dict, pdf_data: Dict) -> ProjectMode:
        """Detekuje mód projektu na základě více kritérií"""
        wav_count = len(project_info['wavs'])
        pdf_result = next(iter(pdf_data.values()), None)
        
        # Pokud PDF extrakce selhala, použijeme základní detekci
        if not pdf_result or pdf_result.get('status') != 'success':
            if wav_count <= 4:
                return ProjectMode.CONSOLIDATED
            return ProjectMode.INDIVIDUAL
            
        pdf_tracks = pdf_result.get('data', [])
        
        # Detekce na základě počtu stran v PDF
        sides = {track.get('side', 'N/A').upper() for track in pdf_tracks if track.get('side')}
        
        # Pokud máme více stran a málo WAV souborů, pravděpodobně jde o konsolidovaný mód
        if len(sides) > 1 and wav_count <= 4:
            return ProjectMode.CONSOLIDATED
            
        # Detekce na základě názvů WAV souborů
        has_numbered_wavs = any(
            re.match(r'^\d{1,2}[_.\s-]', Path(p).name) 
            for p in project_info['wavs']
        )
        
        if not has_numbered_wavs and wav_count <= 4:
            return ProjectMode.CONSOLIDATED
            
        return ProjectMode.INDIVIDUAL

    def _generate_report(self, projects: Dict, pdf_data: Dict, wav_durations: Dict, output_filename: Path) -> None:
        """Generuje report s validací projektů"""
        with open(output_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.config.CSV_HEADERS)
            writer.writeheader()
            
            total_projects = len(projects)
            for i, (project_name, project_info) in enumerate(projects.items()):
                self.status_callback(f"5/5 Validuji projekt {i+1}/{total_projects}: {project_name}")
                self.progress_callback(i, total_projects)
                
                # Detekce módu projektu
                project_pdf_results = {
                    p.as_posix(): pdf_data.get(p.as_posix()) 
                    for p in project_info['pdfs']
                }
                mode = self._detect_project_mode(project_info, project_pdf_results)
                
                project_wav_durations = {
                    p.as_posix(): wav_durations.get(p.as_posix()) 
                    for p in project_info['wavs']
                }

                # Validace podle detekovaného módu
                if mode == ProjectMode.CONSOLIDATED:
                    validation_rows = self._validate_consolidated_project(
                        project_name, project_pdf_results, project_wav_durations
                    )
                else:
                    validation_rows = self._validate_individual_project(
                        project_name, project_pdf_results, project_wav_durations
                    )
                
                # Zápis řádků do CSV
                for row in validation_rows:
                    writer.writerow(row)
                f.flush()

    def _create_validation_row(self, project_name: str, status: str, validation_item: str, 
                              item_type: str, pdf_duration: Optional[float], 
                              wav_duration: Optional[float], pdf_source: str, 
                              wav_source: str, notes: str) -> Dict[str, Any]:
        """Vytvoří standardizovaný řádek pro report"""
        diff = wav_duration - pdf_duration if wav_duration is not None and pdf_duration is not None else None
        
        return {
            "project_title": project_name,
            "status": status,
            "validation_item": validation_item,
            "item_type": item_type,
            "pdf_duration_mmss": seconds_to_mmss(pdf_duration).replace('+', ''),
            "wav_duration_mmss": seconds_to_mmss(wav_duration).replace('+', ''),
            "difference_mmss": seconds_to_mmss(diff),
            "pdf_duration_sec": round(pdf_duration, 2) if pdf_duration is not None else None,
            "wav_duration_sec": round(wav_duration, 2) if wav_duration is not None else None,
            "difference_sec": round(diff, 2) if diff is not None else None,
            "pdf_source": pdf_source,
            "wav_source": wav_source,
            "notes": notes
        }

    def _validate_consolidated_project(self, project_name: str, pdf_results: Dict[str, Dict], 
                                      wav_durations: Dict[str, Optional[float]]) -> List[Dict]:
        """Validuje projekt v konsolidovaném módu"""
        rows = []
        pdf_result = next(iter(pdf_results.values()), None)
        
        if not pdf_result or pdf_result.get('status') != 'success':
            pdf_path_str = next(iter(pdf_results.keys()))
            return [self._create_validation_row(
                project_name=project_name,
                status="FAIL",
                validation_item="",
                item_type="PROJECT",
                pdf_duration=None,
                wav_duration=None,
                pdf_source=Path(pdf_path_str).name,
                wav_source="",
                notes=f"Extrakce dat z PDF selhala: {pdf_result.get('error_message', 'Neznámá chyba')}"
            )]

        pdf_tracks = pdf_result.get('data', [])
        pdf_path_str = pdf_result.get('source_identifier')

        # Seskupení skladeb podle stran
        sides = {}
        for track in pdf_tracks:
            side = str(track.get('side', 'N/A')).upper()
            if side not in sides: 
                sides[side] = []
            sides[side].append(track)

        available_wavs = wav_durations.copy()

        for side, tracks_on_side in sides.items():
            pdf_total_duration = sum(
                t.get('duration_seconds', 0) 
                for t in tracks_on_side 
                if t.get('duration_seconds') is not None
            )

            # Hledání odpovídajícího WAV souboru
            wav_path_for_side = next(
                (p for p in available_wavs 
                 if f"side_{side.lower()}" in Path(p).name.lower() or 
                    f"side {side.lower()}" in Path(p).name.lower()), 
                None
            )
            
            if not wav_path_for_side:
                wav_path_for_side = next(
                    (p for p in available_wavs if "master" in Path(p).name.lower()), 
                    None
                )

            wav_dur = available_wavs.pop(wav_path_for_side, None) if wav_path_for_side else None

            # Validace
            diff = wav_dur - pdf_total_duration if wav_dur is not None else None
            status = "OK"
            notes = f"Celkem {len(tracks_on_side)} skladeb."
            
            if diff is not None and abs(diff) > self.config.VALIDATION_TOLERANCE_SECONDS:
                status = "ERROR"
                notes += f" Rozdíl překročil toleranci {self.config.VALIDATION_TOLERANCE_SECONDS}s."
            elif wav_dur is None:
                status = "FAIL"
                notes = "Nepodařilo se najít odpovídající WAV pro stranu."

            rows.append(self._create_validation_row(
                project_name=project_name,
                status=status,
                validation_item=f"Side {side}",
                item_type="SIDE",
                pdf_duration=pdf_total_duration,
                wav_duration=wav_dur,
                pdf_source=Path(pdf_path_str).name,
                wav_source=Path(wav_path_for_side).name if wav_path_for_side else "N/A",
                notes=notes
            ))
        return rows

    def _validate_individual_project(self, project_name: str, pdf_results: Dict[str, Dict], 
                                   wav_durations: Dict[str, Optional[float]]) -> List[Dict]:
        """Validuje projekt v individuálním módu"""
        rows = []
        pdf_result = next(iter(pdf_results.values()), None)
        
        if not pdf_result or pdf_result.get('status') != 'success':
            pdf_path_str = next(iter(pdf_results.keys()))
            return [self._create_validation_row(
                project_name=project_name,
                status="FAIL",
                validation_item="",
                item_type="PROJECT",
                pdf_duration=None,
                wav_duration=None,
                pdf_source=Path(pdf_path_str).name,
                wav_source="",
                notes=f"Extrakce dat z PDF selhala: {pdf_result.get('error_message', 'Neznámá chyba')}"
            )]

        pdf_tracks = pdf_result.get('data', [])
        pdf_path_str = pdf_result.get('source_identifier')

        available_wavs = {k: v for k, v in wav_durations.items() if v is not None}
        
        for track in pdf_tracks:
            pdf_dur = track.get('duration_seconds')
            track_title = track.get('title', '')

            # Pokus o spárování WAV souboru
            best_match_wav, highest_score = None, 0
            for wav_path, wav_dur in available_wavs.items():
                score = fuzz.token_set_ratio(
                    normalize_string(track_title), 
                    normalize_string(Path(wav_path).stem)
                )
                if score > highest_score:
                    highest_score, best_match_wav = score, wav_path

            # Fallback strategie: párování podle pořadí
            if highest_score < self.config.SIMILARITY_THRESHOLD:
                track_number = track.get('track_number')
                if track_number is not None:
                    for wav_path, wav_dur in available_wavs.items():
                        wav_name = Path(wav_path).stem
                        if re.match(rf'^{track_number}[\s._-]', wav_name):
                            best_match_wav = wav_path
                            highest_score = 100
                            break

            wav_path_str, wav_dur, notes = None, None, ""
            if highest_score >= self.config.SIMILARITY_THRESHOLD:
                wav_path_str, wav_dur = best_match_wav, available_wavs.pop(best_match_wav)
            else:
                notes = "Nepodařilo se spárovat WAV soubor."

            # Validace
            diff = wav_dur - pdf_dur if wav_dur is not None and pdf_dur is not None else None
            status = "OK"
            
            if diff is not None and abs(diff) > self.config.VALIDATION_TOLERANCE_SECONDS:
                status, notes = "ERROR", f"Rozdíl překročil toleranci {self.config.VALIDATION_TOLERANCE_SECONDS}s"
            elif wav_path_str is None:
                status = "FAIL"

            rows.append(self._create_validation_row(
                project_name=project_name,
                status=status,
                validation_item=track_title,
                item_type="TRACK",
                pdf_duration=pdf_dur,
                wav_duration=wav_dur,
                pdf_source=Path(pdf_path_str).name,
                wav_source=Path(wav_path_str).name if wav_path_str else "N/A",
                notes=notes
            ))

        # Zpracování nespárovaných WAV souborů
        for wav_path_str, wav_dur in available_wavs.items():
            rows.append(self._create_validation_row(
                project_name=project_name,
                status="WARN",
                validation_item="",
                item_type="TRACK",
                pdf_duration=None,
                wav_duration=wav_dur,
                pdf_source="",
                wav_source=Path(wav_path_str).name,
                notes="Tento WAV soubor nebyl spárován s žádnou skladbou z PDF."
            ))

        return rows

# ==============================================================================
# --- GRAFICKÉ UŽIVATELSKÉ ROZHRANÍ (GUI) ---
# ==============================================================================

class VinylPreflightApp:
    """Hlavní GUI aplikace"""
    
    def __init__(self, root: tk.Tk, api_key: str):
        self.root = root
        self.api_key = api_key
        self.config = Config()
        self.processor_thread = None
        
        root.title("Vinyl Preflight Processor v3.0")
        root.geometry("800x400")
        root.minsize(600, 300)

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill="both", expand=True)

        input_frame = ttk.LabelFrame(main_frame, text="1. Výběr zdroje", padding="10")
        input_frame.pack(fill="x", pady=5)
        
        self.folder_path = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.folder_path, state="readonly").pack(
            side="left", fill="x", expand=True, padx=(0, 5)
        )
        ttk.Button(input_frame, text="Procházet...", command=self.browse_directory).pack(side="left")

        run_frame = ttk.LabelFrame(main_frame, text="2. Zpracování", padding="10")
        run_frame.pack(fill="x", pady=5)

        self.start_button = ttk.Button(
            run_frame, text="Spustit zpracování", 
            command=self.start_processing, state="disabled"
        )
        self.start_button.pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(
            run_frame, orient="horizontal", mode="determinate"
        )
        self.progress_bar.pack(fill="x", pady=5)
        
        self.status_label = ttk.Label(
            run_frame, text="Připraveno. Vyberte adresář s projekty."
        )
        self.status_label.pack(pady=5)

    def browse_directory(self) -> None:
        """Výběr zdrojového adresáře"""
        directory = filedialog.askdirectory(title="Vyberte kořenový adresář s projekty")
        if directory:
            self.folder_path.set(directory)
            self.start_button.config(state="normal")
            self.status_label.config(text=f"Vybrán adresář: {directory}")

    def start_processing(self) -> None:
        """Spustí zpracování v samostatném vlákně"""
        source_dir = self.folder_path.get()
        if not source_dir or not os.path.isdir(source_dir):
            messagebox.showerror("Chyba", "Vyberte prosím platný adresář.")
            return

        self.start_button.config(state="disabled")
        
        processor = PreflightProcessor(
            self.api_key, self.config, 
            self.update_progress, self.update_status
        )
        self.processor_thread = threading.Thread(
            target=processor.run, args=(source_dir,), daemon=True
        )
        self.processor_thread.start()

    def update_progress(self, value: int, maximum: int) -> None:
        """Aktualizace progress baru"""
        self.root.after(0, self._do_update_progress, value, maximum)

    def _do_update_progress(self, value: int, maximum: int) -> None:
        """Interní aktualizace progress baru"""
        if maximum > 0:
            self.progress_bar["maximum"] = maximum
            self.progress_bar["value"] = value

    def update_status(self, text: str) -> None:
        """Aktualizace statusu"""
        self.root.after(0, self._do_update_status, text)

    def _do_update_status(self, text: str) -> None:
        """Interní aktualizace statusu"""
        self.status_label.config(text=text)

# ==============================================================================
# --- VSTUPNÍ BOD APLIKACE ---
# ==============================================================================

if __name__ == "__main__":
    if os.name == 'nt' or sys.platform == 'darwin':
        mp.freeze_support()
        
    dotenv_path = Path(__file__).resolve().parent.parent / '.env'
    load_dotenv(dotenv_path=dotenv_path)
    API_KEY = os.getenv("OPENROUTER_API_KEY")

    if not API_KEY:
        messagebox.showerror(
            "Chyba konfigurace",
            "API klíč (OPENROUTER_API_KEY) nebyl nalezen.\n\n"
            "Ujistěte se, že máte v hlavní složce projektu soubor '.env' se správným obsahem."
        )
        sys.exit(1)

    root = tk.Tk()
    app = VinylPreflightApp(root, API_KEY)
    root.mainloop()