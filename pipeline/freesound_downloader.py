"""
Freesound.org scraper for the Dino-VAE project.
Downloads crocodilian, alligator, and elephant infrasound — taxa that are
absent from Xeno-Canto (which only covers birds/bats/grasshoppers).

Get a free API key at: https://freesound.org/api/apply/
Add it to .env as:  FREESOUND_API_KEY=your_key_here
"""

import os
import time
import requests
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

FREESOUND_BASE = "https://freesound.org/apiv2"


class FreesoundDownloader:
    def __init__(self, api_key, output_dir="./DATA/birds"):
        if not api_key:
            raise ValueError(
                "FREESOUND_API_KEY missing. Register at freesound.org/api/apply/ "
                "and add it to your .env file."
            )
        self.api_key    = api_key
        self.output_dir = output_dir
        self.session    = requests.Session()
        self.session.update_headers = lambda **kw: None   # suppress DeprecationWarning
        self.session.headers.update({"Authorization": f"Token {api_key}"})

        # ── Canonical folder per search group ─────────────────────────────────
        # All crocodilian queries funnel into the shared Crocodylia/ directory so
        # that the class-centroid calculation sees a single, self-consistent set.
        self.targets = {
            # folder_name : list of (query_string, high_attack_only)
            "Crocodylia": [
                "crocodile call",
                "crocodile vocalization",
                "crocodile bellow",
                "crocodilian sound",
                "alligator bellow",          # percussive, high-attack
                "alligator call",
                "alligator hiss",
                "american alligator",
                "caiman call",
                "caiman sound",
                "crocodile jaw snap",         # high-attack transient target
                "alligator headslap",         # resonant boom + splash
                "alligator infrasound",
                "gharial call",
                "mugger crocodile",
                "saltwater crocodile",
                "nile crocodile",
            ],
            "Elephantidae": [
                "elephant infrasound",
                "elephant rumble",
                "elephant call",
                "elephant trumpet",
                "elephant contact call",
            ],
        }

    def _search(self, query, page=1, page_size=150):
        """Query Freesound text search and return results list."""
        params = {
            "query":      query,
            "page":       page,
            "page_size":  page_size,
            "fields":     "id,name,previews,license,duration",
            "filter":     "duration:[3 TO 300]",   # 3 s – 5 min clips only
        }
        resp = self.session.get(f"{FREESOUND_BASE}/search/text/", params=params, timeout=30)
        if resp.status_code != 200:
            print(f"    [warn] HTTP {resp.status_code} for query '{query}'")
            return []
        return resp.json().get("results", [])

    def _download_file(self, sound_id, preview_url, save_path):
        """Download the HQ MP3 preview for a Freesound entry."""
        try:
            r = self.session.get(preview_url, timeout=60, stream=True)
            r.raise_for_status()
            with open(save_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        except Exception as e:
            print(f"    [err] {save_path}: {e}")
            return False

    def download_data(self, target_per_folder: int = 500):
        for folder, queries in self.targets.items():
            print(f"\n{'='*55}")
            print(f"Freesound — {folder}  (target {target_per_folder})")

            species_dir = os.path.join(self.output_dir, folder)
            os.makedirs(species_dir, exist_ok=True)

            # Build seen-ID set from filename stems to avoid re-downloading
            seen_ids: set[int] = set()
            for f in os.listdir(species_dir):
                parts = os.path.splitext(f)[0].split("_")
                try:
                    seen_ids.add(int(parts[-1]))
                except (ValueError, IndexError):
                    pass

            existing    = len([f for f in os.listdir(species_dir) if f.endswith(".mp3")])
            total_saved = 0

            for query in queries:
                if existing + total_saved >= target_per_folder:
                    break
                print(f"  Query: '{query}'")
                results = self._search(query)
                print(f"  → {len(results)} results")

                for sound in results:
                    if existing + total_saved >= target_per_folder:
                        break
                    sid = sound["id"]
                    if sid in seen_ids:
                        continue
                    seen_ids.add(sid)

                    preview_url = sound.get("previews", {}).get("preview-hq-mp3")
                    if not preview_url:
                        continue

                    fname     = f"{folder}_{sid}.mp3"
                    save_path = os.path.join(species_dir, fname)

                    if os.path.exists(save_path):
                        continue

                    print(f"    [dl]  {fname}  ({sound.get('duration', 0):.1f}s)")
                    if self._download_file(sid, preview_url, save_path):
                        total_saved += 1
                    time.sleep(0.5)

            print(f"  → {total_saved} new files saved to {folder}/")

        print("\n\nFreesound download complete!")
        print("\nSummary:")
        for folder in self.targets:
            d = os.path.join(self.output_dir, folder)
            if os.path.isdir(d):
                n = len([f for f in os.listdir(d) if f.endswith(".mp3")])
                print(f"  {folder:<20}: {n} files")


if __name__ == "__main__":
    API_KEY = os.getenv("FREESOUND_API_KEY", "")
    try:
        dl = FreesoundDownloader(api_key=API_KEY, output_dir="./DATA/birds")
        dl.download_data(target_per_folder=500)
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
