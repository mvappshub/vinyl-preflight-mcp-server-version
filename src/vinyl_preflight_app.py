#!/usr/bin/env python3
"""
Vinyl Preflight Processor v2.5 - Finální Produkční Verze

- Implementuje plnou podporu pro "Consolidated Side" mód (jeden WAV na stranu).
- Automaticky detekuje mód projektu a volí správnou validační strategii.
- Poskytuje přesné a relevantní reporty pro všechny typy projektů.
"""
# ... (všechny importy a konfigurace zůstávají stejné) ...
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
from typing import Callable, Dict, List, Tuple, Optional
import shutil
import tempfile
import zipfile
import re

import requests
import soundfile as sf
from dotenv import load_dotenv
import multiprocessing as mp
import fitz
from thefuzz import fuzz
import logging

# Configure logging for this module
logger = logging.getLogger(__name__)

# Inicializace rarfile s kontrolou dostupnosti
rarfile = None  # Inicializace proměnné
try:
    import rarfile
    if not shutil.which("unrar"):
        logger.warning("Příkaz 'unrar' nebyl nalezen v systémové PATH. Extrakce RAR nemusí fungovat.")
except ImportError:
    logger.warning("Knihovna 'rarfile' není nainstalována. Podpora pro .rar archivy je vypnuta.")

MODEL_NAME = "google/gemini-2.5-flash"
API_URL = "https://openrouter.ai/api/v1/chat/completions"
MAX_PDFS_PER_BATCH = 50
MAX_PARALLEL_API_REQUESTS = 10
API_REQUEST_TIMEOUT = 180
VALIDATION_TOLERANCE_SECONDS = 10
CSV_HEADERS = [
    "project_title", "status", "validation_item", "item_type",
    "pdf_duration_mmss", "wav_duration_mmss", "difference_mmss",
    "pdf_duration_sec", "wav_duration_sec", "difference_sec",
    "pdf_source", "wav_source", "notes"
]

def seconds_to_mmss(seconds: Optional[float]) -> str:
    if seconds is None: return "N/A"
    if not isinstance(seconds, (int, float)): return "Chyba"
    sign = '-' if seconds < 0 else '+'
    seconds = abs(seconds)
    minutes, remaining_seconds = divmod(round(seconds), 60)
    return f"{sign}{minutes:02d}:{remaining_seconds:02d}"

def _get_wav_duration_worker(filepath: Path) -> Tuple[str, Optional[float]]:
    try:
        info = sf.info(filepath)
        return filepath.as_posix(), info.duration
    except Exception as e:
        logger.error(f"Chyba při čtení WAV '{filepath.name}': {e}")
        return filepath.as_posix(), None

def normalize_string(s: str) -> str:
    s = s.lower()
    s = re.sub(r'^\d+[\s.-]*', '', s)
    s = re.sub(r'[\W_]+', ' ', s)
    return " ".join(s.split())

class PreflightProcessor:
    # ... (metody __init__, run, a všechny pomocné metody až po _validate_project
    # zůstávají stejné jako ve verzi 2.4) ...
    def __init__(self, api_key: str, progress_callback: Callable, status_callback: Callable):
        if not api_key: raise ValueError("API klíč nesmí být prázdný.")
        self.api_key = api_key
        self.progress_callback = progress_callback
        self.status_callback = status_callback
        self.headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def run(self, source_directory: str):
        try:
            start_time = time.time()
            output_dir = Path(__file__).resolve().parent.parent / "output"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_filename = output_dir / f"Preflight_Report_{time.strftime('%Y-%m-%d_%H-%M-%S')}.csv"

            with tempfile.TemporaryDirectory(prefix="preflight_") as tmpdir:
                temp_path = Path(tmpdir)
                self.status_callback("1/5 Připravuji pracovní prostor a extrahuji archivy...")
                self._prepare_workspace(Path(source_directory), temp_path)

                self.status_callback("2/5 Skenuji soubory a připravuji projekty...")
                projects = self._scan_and_group_projects(temp_path)
                if not projects:
                    self.status_callback("Ve vybraném adresáři (včetně archivů) nebyly nalezeny žádné relevantní podsložky s PDF a WAV soubory.")
                    self.status_callback("Připraveno.")
                    return None

                self.status_callback("3/5 Zjišťuji délky WAV souborů...")
                wav_durations = self._get_all_wav_durations(projects)

                self.status_callback("4/5 Vytvářím dávky PDF pro efektivní extrakci...")
                pdf_batches = self._create_pdf_batches(projects)
                
                self.status_callback(f"4/5 Budu zpracovávat {len(pdf_batches)} dávek PDF. Odesílám k LLM...")
                extracted_pdf_data = self._process_all_pdf_batches(pdf_batches)
                
                self.status_callback("5/5 Zahajuji finální validaci a zápis do reportu...")
                with open(output_filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                    writer.writeheader()
                    
                    total_projects = len(projects)
                    for i, (project_name, project_info) in enumerate(projects.items()):
                        self.status_callback(f"5/5 Validuji projekt {i+1}/{total_projects}: {project_name}")
                        self.progress_callback(i, total_projects)
                        
                        project_pdf_results = {p.as_posix(): extracted_pdf_data.get(p.as_posix()) for p in project_info['pdfs']}
                        project_wav_durations = {p.as_posix(): wav_durations.get(p.as_posix()) for p in project_info['wavs']}

                        validation_rows = self._validate_project(project_name, project_pdf_results, project_wav_durations)
                        
                        for row in validation_rows:
                            writer.writerow(row)
                        f.flush()
                
                self.progress_callback(total_projects, total_projects)

            end_time = time.time()
            self.status_callback(f"Hotovo! Celkový čas: {end_time - start_time:.2f} s. Report uložen do: {output_filename}")
            return str(output_filename)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_callback(f"Chyba: Proces byl přerušen. {e}")
            return None

    def _prepare_workspace(self, source_root: Path, temp_root: Path):
        for item in source_root.iterdir():
            if item.is_dir():
                shutil.copytree(item, temp_root / item.name, dirs_exist_ok=True)
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

    def _scan_and_group_projects(self, root_dir: Path) -> Dict[str, Dict[str, List[Path]]]:
        projects = {}
        for item in root_dir.iterdir():
            if item.is_dir():
                pdfs = list(item.rglob("*.pdf"))
                wavs = list(item.rglob("*.wav"))
                if pdfs and wavs:
                    projects[item.name] = {'pdfs': pdfs, 'wavs': wavs}
        return projects

    def _get_all_wav_durations(self, projects: dict) -> Dict[str, Optional[float]]:
        all_wav_paths = [wav for proj in projects.values() for wav in proj['wavs']]
        durations = {}
        with mp.Pool() as pool:
            results = pool.map(_get_wav_duration_worker, all_wav_paths)
        for path_str, duration in results:
            durations[path_str] = duration
        return durations

    def _create_pdf_batches(self, projects: dict) -> List[List[Path]]:
        all_pdfs_to_process = [pdf_path for proj in projects.values() for pdf_path in proj['pdfs']]
        batches = []
        for i in range(0, len(all_pdfs_to_process), MAX_PDFS_PER_BATCH):
            batches.append(all_pdfs_to_process[i:i + MAX_PDFS_PER_BATCH])
        return batches

    def _process_all_pdf_batches(self, batches: list) -> dict:
        all_results = {}
        total_batches = len(batches)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PARALLEL_API_REQUESTS) as executor:
            future_to_batch = {executor.submit(self._process_single_extraction_batch, batch): i for i, batch in enumerate(batches)}
            
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

    def _process_single_extraction_batch(self, batch: List[Path]) -> Optional[List[Dict]]:
        documents_to_process = []
        for pdf_path in batch:
            try:
                with fitz.open(pdf_path) as doc:
                    text = "".join(page.get_text() for page in doc)
                if not text.strip():
                    text = f"VAROVÁNÍ: PDF soubor '{pdf_path.name}' neobsahuje žádný extrahovatelný text."
                
                documents_to_process.append({"identifier": pdf_path.as_posix(), "content": text})
            except Exception as e:
                 logger.error(f"Chyba při čtení PDF pro dávku: {pdf_path}, {e}")
                 documents_to_process.append({"identifier": pdf_path.as_posix(), "content": f"CHYBA: Nelze přečíst soubor. {e}"})

        if not documents_to_process: return None

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
        payload = { "model": MODEL_NAME, "messages": [{"role": "user", "content": prompt}], "response_format": {"type": "json_object"}, "temperature": 0.0 }
        
        try:
            response = requests.post(API_URL, headers=self.headers, json=payload, timeout=API_REQUEST_TIMEOUT)
            response.raise_for_status()
            content_str = response.json()["choices"][0]["message"]["content"]
            return json.loads(content_str).get("results", [])
        except Exception as e:
            logger.error(f"Chyba API volání pro dávku: {e}")
            return [{"source_identifier": d["identifier"], "status": "error", "data": [], "error_message": str(e)} for d in documents_to_process]

    def _validate_project(self, project_name: str, pdf_results: Dict[str, Dict], wav_durations: Dict[str, Optional[float]]) -> List[Dict]:
        """
        Hlavní validační metoda, která funguje jako "dispečer".
        Na základě předem určeného módu volá správnou validační funkci.
        """
        # --- ZPŘESNĚNÁ DETEKCE MÓDU ---
        wav_paths = list(wav_durations.keys())
        wav_count = len(wav_paths)
        is_consolidated = (wav_count > 0 and wav_count <= 4 and
                           all(not re.match(r'^\d{1,2}[_.\s-]', Path(p).name) for p in wav_paths))

        if is_consolidated:
            return self._validate_consolidated_project(project_name, pdf_results, wav_durations)
        else:
            return self._validate_individual_project(project_name, pdf_results, wav_durations)

    def _validate_consolidated_project(self, project_name: str, pdf_results: Dict[str, Dict], wav_durations: Dict[str, Optional[float]]) -> List[Dict]:
        """Zpracovává POUZE projekty v konsolidovaném módu."""
        rows = []
        pdf_result = next(iter(pdf_results.values()), None)
        if not pdf_result or pdf_result.get('status') != 'success':
            pdf_path_str = next(iter(pdf_results.keys()))
            rows.append({"project_title": project_name, "status": "FAIL", "pdf_source": Path(pdf_path_str).name, "notes": f"Extrakce dat z PDF selhala: {pdf_result.get('error_message', 'Neznámá chyba')}"})
            return rows

        pdf_tracks = pdf_result.get('data', [])
        pdf_path_str = pdf_result.get('source_identifier')

        sides = {}
        for track in pdf_tracks:
            side = str(track.get('side', 'N/A')).upper()
            if side not in sides: sides[side] = []
            sides[side].append(track)

        available_wavs = wav_durations.copy()

        for side, tracks_on_side in sides.items():
            pdf_total_duration = sum(t.get('duration_seconds', 0) for t in tracks_on_side if t.get('duration_seconds') is not None)

            wav_path_for_side = next((p for p in available_wavs if f"side_{side.lower()}" in Path(p).name.lower() or f"side {side.lower()}" in Path(p).name.lower()), None)
            if not wav_path_for_side:
                wav_path_for_side = next((p for p in available_wavs if "master" in Path(p).name.lower()), None)

            wav_dur = available_wavs.pop(wav_path_for_side, None) if wav_path_for_side else None

            diff = wav_dur - pdf_total_duration if wav_dur is not None else None
            status = "OK"
            notes = f"Celkem {len(tracks_on_side)} skladeb."
            if diff is not None and abs(diff) > VALIDATION_TOLERANCE_SECONDS:
                status = "ERROR"
                notes += f" Rozdíl překročil toleranci {VALIDATION_TOLERANCE_SECONDS}s."
            elif wav_dur is None:
                status = "FAIL"
                notes = "Nepodařilo se najít odpovídající WAV pro stranu."

            rows.append({
                "project_title": project_name, "status": status, "validation_item": f"Side {side}",
                "item_type": "SIDE", "pdf_duration_mmss": seconds_to_mmss(pdf_total_duration).replace('+', ''),
                "wav_duration_mmss": seconds_to_mmss(wav_dur).replace('+', ''), "difference_mmss": seconds_to_mmss(diff),
                # --- ZDE JE ZMĚNA: ZAOKROUHLENÍ ---
                "pdf_duration_sec": round(pdf_total_duration, 2),
                "wav_duration_sec": round(wav_dur, 2) if wav_dur is not None else None,
                "difference_sec": round(diff, 2) if diff is not None else None,
                # --- KONEC ZMĚNY ---
                "pdf_source": Path(pdf_path_str).name,
                "wav_source": Path(wav_path_for_side).name if wav_path_for_side else "N/A",
                "notes": notes
            })
        return rows

    def _validate_individual_project(self, project_name: str, pdf_results: Dict[str, Dict], wav_durations: Dict[str, Optional[float]]) -> List[Dict]:
        """Zpracovává POUZE projekty v individuálním módu."""
        rows = []
        pdf_result = next(iter(pdf_results.values()), None)
        if not pdf_result or pdf_result.get('status') != 'success':
            pdf_path_str = next(iter(pdf_results.keys()))
            rows.append({"project_title": project_name, "status": "FAIL", "pdf_source": Path(pdf_path_str).name, "notes": f"Extrakce dat z PDF selhala: {pdf_result.get('error_message', 'Neznámá chyba')}"})
            return rows

        pdf_tracks = pdf_result.get('data', [])
        pdf_path_str = pdf_result.get('source_identifier')

        available_wavs = {k: v for k, v in wav_durations.items() if v is not None}
        for track in pdf_tracks:
            pdf_dur = track.get('duration_seconds')
            track_title = track.get('title', '')

            best_match_wav, highest_score = None, 0
            for wav_path, wav_dur in available_wavs.items():
                score = fuzz.token_set_ratio(normalize_string(track_title), normalize_string(Path(wav_path).stem))
                if score > highest_score:
                    highest_score, best_match_wav = score, wav_path

            wav_path_str, wav_dur, notes = None, None, ""
            if highest_score > 70:
                wav_path_str, wav_dur = best_match_wav, available_wavs.pop(best_match_wav)
            else:
                notes = "Nepodařilo se spárovat WAV soubor podle názvu."

            diff = wav_dur - pdf_dur if wav_dur is not None and pdf_dur is not None else None
            status = "OK"
            if diff is not None and abs(diff) > VALIDATION_TOLERANCE_SECONDS:
                status, notes = "ERROR", f"Rozdíl překročil toleranci {VALIDATION_TOLERANCE_SECONDS}s"
            elif wav_path_str is None:
                status = "FAIL"

            rows.append({
                "project_title": project_name, "status": status, "validation_item": track_title,
                "item_type": "TRACK", "pdf_duration_mmss": seconds_to_mmss(pdf_dur).replace('+', ''),
                "wav_duration_mmss": seconds_to_mmss(wav_dur).replace('+', ''), "difference_mmss": seconds_to_mmss(diff),
                # --- ZDE JE ZMĚNA: ZAOKROUHLENÍ ---
                "pdf_duration_sec": round(pdf_dur, 2) if pdf_dur is not None else None,
                "wav_duration_sec": round(wav_dur, 2) if wav_dur is not None else None,
                "difference_sec": round(diff, 2) if diff is not None else None,
                # --- KONEC ZMĚNY ---
                "pdf_source": Path(pdf_path_str).name,
                "wav_source": Path(wav_path_str).name if wav_path_str else "N/A",
                "notes": notes
            })

        for wav_path_str, wav_dur in available_wavs.items():
            rows.append({"project_title": project_name, "status": "WARN", "wav_source": Path(wav_path_str).name, "notes": "Tento WAV soubor nebyl spárován s žádnou skladbou z PDF."})

        return rows

# ==============================================================================
# --- GRAFICKÉ UŽIVATELSKÉ ROZHRANÍ (GUI) ---
# ==============================================================================
class VinylPreflightApp:
    def __init__(self, root: tk.Tk, api_key: str):
        self.root = root
        self.api_key = api_key
        self.processor_thread = None
        root.title("Vinyl Preflight Processor v2.5")
        root.geometry("800x400")
        root.minsize(600, 300)

        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill="both", expand=True)

        input_frame = ttk.LabelFrame(main_frame, text="1. Výběr zdroje", padding="10")
        input_frame.pack(fill="x", pady=5)
        
        self.folder_path = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.folder_path, state="readonly").pack(side="left", fill="x", expand=True, padx=(0, 5))
        ttk.Button(input_frame, text="Procházet...", command=self.browse_directory).pack(side="left")

        run_frame = ttk.LabelFrame(main_frame, text="2. Zpracování", padding="10")
        run_frame.pack(fill="x", pady=5)

        self.start_button = ttk.Button(run_frame, text="Spustit zpracování", command=self.start_processing, state="disabled")
        self.start_button.pack(pady=10)
        
        self.progress_bar = ttk.Progressbar(run_frame, orient="horizontal", mode="determinate")
        self.progress_bar.pack(fill="x", pady=5)
        
        self.status_label = ttk.Label(run_frame, text="Připraveno. Vyberte adresář s projekty.")
        self.status_label.pack(pady=5)

    def browse_directory(self):
        directory = filedialog.askdirectory(title="Vyberte kořenový adresář s projekty")
        if directory:
            self.folder_path.set(directory)
            self.start_button.config(state="normal")
            self.status_label.config(text=f"Vybrán adresář: {directory}")

    def start_processing(self):
        source_dir = self.folder_path.get()
        if not source_dir or not os.path.isdir(source_dir):
            messagebox.showerror("Chyba", "Vyberte prosím platný adresář.")
            return

        self.start_button.config(state="disabled")
        
        processor = PreflightProcessor(self.api_key, self.update_progress, self.update_status)
        self.processor_thread = threading.Thread(target=processor.run, args=(source_dir,), daemon=True)
        self.processor_thread.start()

    def update_progress(self, value: int, maximum: int):
        self.root.after(0, self._do_update_progress, value, maximum)

    def _do_update_progress(self, value: int, maximum: int):
        if maximum > 0:
            self.progress_bar["maximum"] = maximum
            self.progress_bar["value"] = value

    def update_status(self, text: str):
        self.root.after(0, self._do_update_status, text)

    def _do_update_status(self, text: str):
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