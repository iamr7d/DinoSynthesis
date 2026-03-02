import os
import requests
import time
from urllib.request import urlretrieve
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


class RatiteDownloader:
    """
    Downloads bioacoustic audio from the Xeno-Canto API v3 for the Dino-VAE project.
    Targets all Archosaur + mass-anchor groups at relaxed quality (q:">C" = A, B, or C).
    """
    def __init__(self, api_key, output_dir="./DATA/birds"):
        if not api_key or api_key == "YOUR_API_KEY":
            raise ValueError("Xeno-Canto API v3 requires a valid API key. Get yours at xeno-canto.org/account")

        self.api_key  = api_key
        self.output_dir  = output_dir
        self.api_base = "https://xeno-canto.org/api/3/recordings"

        # Expanded taxonomy targets
        # Ratites + Tinamous (sister clade, phylogenetically critical) + large deep-calling birds
        # q:">D" includes A/B/C/D — maximum volume, only excludes unrateable recordings
        self.targets = {
            # ── Ratites ────────────────────────────────────────────────────────
            "Cassowary":   'gen:Casuarius q:">D"',
            "Emu":         'gen:Dromaius q:">D"',
            "Ostrich":     'gen:Struthio q:">D"',
            "Rhea":        'gen:Rhea q:">D"',
            "Kiwi":        'gen:Apteryx q:">D"',

            # ── Tinamous (sister clade to all ratites — key for interpolation) ─
            "Tinamou_Tinamus":      'gen:Tinamus q:">D"',       # Great, White-throated Tinamous
            "Tinamou_Crypturellus": 'gen:Crypturellus q:">D"',  # Largest genus, ~20 spp.
            "Tinamou_Rhynchotus":   'gen:Rhynchotus q:">D"',    # Red-winged Tinamou
            "Tinamou_Nothura":      'gen:Nothura q:">D"',
            "Tinamou_Eudromia":     'gen:Eudromia q:">D"',      # Elegant Crested
            "Tinamou_Nothoprocta":  'gen:Nothoprocta q:">D"',   # Andean Tinamous
            "Tinamou_Tinamotis":    'gen:Tinamotis q:">D"',
            "Tinamou_Nothocercus":  'gen:Nothocercus q:">D"',
            "Tinamou_Taoniscus":    'gen:Taoniscus q:">D"',

            # ── Large bustards (extreme body-mass anchors, deep resonance) ────
            "Bustard_Ardeotis":  'gen:Ardeotis q:">D"',   # Kori/Great Indian Bustard
            "Bustard_Otis":      'gen:Otis q:">D"',        # Great Bustard
            "Bustard_Neotis":    'gen:Neotis q:">D"',

            # ── Deep-booming wading birds (acoustic signature anchors) ────────
            "Shoebill":    'gen:Balaeniceps q:">D"',
            "Bittern":     'gen:Botaurus q:">D"',          # Booming bitterns
            "Sunbittern":  'gen:Eurypyga q:">D"',
        }

    def _fetch_all_pages(self, query):
        """Paginates through all results for a query and returns every recording."""
        recordings = []
        page = 1
        while True:
            params = {
                "query":    query,
                "key":      self.api_key,
                "per_page": 500,
                "page":     page,
            }
            resp = requests.get(self.api_base, params=params, timeout=30)
            if resp.status_code != 200:
                try:
                    err = resp.json().get("error", {}).get("message", "unknown error")
                except Exception:
                    err = resp.text
                print(f"    API error (HTTP {resp.status_code}): {err}")
                break

            data       = resp.json()
            page_recs  = data.get("recordings", [])
            recordings.extend(page_recs)

            num_pages = int(data.get("numPages", 1))
            print(f"    Page {page}/{num_pages} — {len(page_recs)} recordings")
            if page >= num_pages:
                break
            page += 1
            time.sleep(0.5)  # polite inter-page delay

        return recordings

    def download_data(self):
        for species, query in self.targets.items():
            print(f"\n{'='*50}")
            print(f"Searching Xeno-Canto for: {species}  (query: {query})")

            recordings = self._fetch_all_pages(query)
            total = len(recordings)
            print(f"  → {total} recordings found. Starting download...")

            if total == 0:
                print("  No recordings found. Skipping.")
                continue

            species_dir = os.path.join(self.output_dir, species)
            os.makedirs(species_dir, exist_ok=True)

            for rec in recordings:
                raw_url   = rec.get("file", "")
                file_url  = ("https:" + raw_url) if raw_url.startswith("//") else raw_url
                file_name = f"{species}_{rec.get('id')}.mp3"
                save_path = os.path.join(species_dir, file_name)

                if os.path.exists(save_path):
                    print(f"  [skip] {file_name} already exists.")
                    continue

                if not file_url:
                    print(f"  [warn] No file URL for recording {rec.get('id')}. Skipping.")
                    continue

                try:
                    print(f"  [dl]   {file_name}  (quality: {rec.get('q','?')}, length: {rec.get('length','?')})")
                    urlretrieve(file_url, save_path)
                    time.sleep(0.75)  # polite download delay
                except Exception as e:
                    print(f"  [err]  {file_name}: {e}")

        print("\n\nAll downloads complete!")
        print("\nDataset summary:")
        for species in self.targets:
            species_dir = os.path.join(self.output_dir, species)
            if os.path.isdir(species_dir):
                count = len([f for f in os.listdir(species_dir) if f.endswith(".mp3")])
                print(f"  {species:<15}: {count} files")


if __name__ == "__main__":
    API_KEY = os.getenv("XC_API_KEY", "")

    print("Initializing Dino-VAE Data Pipeline (Expanded Download)...")
    try:
        downloader = RatiteDownloader(api_key=API_KEY, output_dir="./DATA/birds")
        downloader.download_data()
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
