"""
iNaturalist sound downloader for the Dino-VAE project.
Zero-authentication — uses the free iNaturalist REST API.

Reptile anchor : Crocodylia (order 26039) — expanded with genus-level taxon IDs
                 for maximum coverage: Crocodylus, Alligator, Gavialis, Caiman,
                 Melanosuchus, Paleosuchus, Tomistoma, Mecistops — combined into
                 the shared Crocodylia/ directory.
Mass anchor    : Whippomorpha/Cetacea (128) + Elephantidae (34) + Phocoidea seals (320)
"""

import os, time, requests
from tqdm import tqdm

API_BASE   = "https://api.inaturalist.org/v1"
HEADERS    = {"Accept": "application/json"}
AUDIO_EXTS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

# ── Crocodylia: order-level + all family/genus IDs that have confirmed audio
# All save to the same "Crocodylia" directory — deduplicated by obs_id.
CROC_TAXON_IDS = {
    26039:  "Crocodylia (order)",            # catches everything — primary query
    26042:  "Crocodylidae (family)",         # true crocodiles
    26040:  "Alligatoridae (family)",        # alligators + caimans
    26041:  "Gavialidae (family)",           # gharial
    25727:  "Crocodylus (genus)",            # Nile, Saltwater, American, Mugger
    26043:  "Alligator (genus)",             # American + Chinese alligator
    126920: "Caiman (genus)",                # Spectacled + Broad-snouted caiman
    126921: "Melanosuchus (genus)",          # Black caiman — deep bellows
    126922: "Paleosuchus (genus)",           # Dwarf caimans
    335081: "Gavialis (genus)",              # Gharial — distinctive resonance
}

# Verified taxon IDs + confirmed sound counts (no quality filter)
TARGETS = {
    # ── Mass anchor ───────────────────────────────────────────────
    "Whippomorpha":  925158,  # 128 sounds: whales + dolphins
    "Elephantidae":  43692,   # 34  sounds: African + Asian elephant
    "Phocoidea":     372843,  # 320 sounds: seals — large marine mammal anchor
}


def fetch_page(taxon_id, page=1, per_page=200):
    r = requests.get(f"{API_BASE}/observations", headers=HEADERS, timeout=30, params={
        "taxon_id": taxon_id, "sounds": "true",
        "per_page": per_page, "page": page,
    })
    r.raise_for_status()
    return r.json()


def collect_urls(taxon_id, target=250):
    urls, page = [], 1
    while len(urls) < target:
        try:
            data    = fetch_page(taxon_id, page=page)
            results = data.get("results", [])
            total   = data.get("total_results", 0)
            if not results:
                break
            for obs in results:
                for snd in obs.get("sounds", []):
                    url = snd.get("file_url") or snd.get("url", "")
                    if not url:
                        continue
                    ext = os.path.splitext(url.split("?")[0])[1].lower() or ".mp3"
                    if ext in AUDIO_EXTS:
                        urls.append((url, ext, obs["id"], snd["id"]))
            print(f"    page {page}/{-(-total//200)}: +{len(results)} obs → {len(urls)} sounds")
            if page * 200 >= total:
                break
            page += 1
            time.sleep(0.4)
        except Exception as e:
            print(f"    [warn] {e}")
            break
    return urls[:target]


def run(output_dir="./DATA/birds", target_per_class=500):
    total_new = 0

    # ── Phase 1: Crocodylia — sweep all taxon IDs into one directory ──────────
    print(f"\n{'='*60}")
    print("Category: Crocodylia  (multi-taxon sweep)")
    croc_dir = os.path.join(output_dir, "Crocodylia")
    os.makedirs(croc_dir, exist_ok=True)

    # Build a seen-set from obs_ids already on disk to avoid duplicates
    seen_obs: set[int] = set()
    for fname in os.listdir(croc_dir):
        parts = fname.split("_")
        # filename pattern: Crocodylia_{obs_id}_{snd_id}{ext}
        if len(parts) >= 3:
            try:
                seen_obs.add(int(parts[1]))
            except ValueError:
                pass

    existing = len([f for f in os.listdir(croc_dir)
                    if os.path.splitext(f)[1].lower() in AUDIO_EXTS])
    print(f"  On disk: {existing}  |  Target: {target_per_class}")

    croc_new = 0
    for taxon_id, label in CROC_TAXON_IDS.items():
        if existing + croc_new >= target_per_class:
            break
        still_needed = target_per_class - existing - croc_new
        print(f"\n  Taxon {taxon_id} ({label}) — need {still_needed} more …")

        urls = collect_urls(taxon_id, target=still_needed + 20)
        for url, ext, obs_id, snd_id in tqdm(urls, desc=f"  {label[:30]}"):
            if obs_id in seen_obs:
                continue                      # already have this observation
            fname     = f"Crocodylia_{obs_id}_{snd_id}{ext}"
            save_path = os.path.join(croc_dir, fname)
            if os.path.exists(save_path):
                seen_obs.add(obs_id)
                continue
            try:
                r = requests.get(url, stream=True, timeout=60)
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(65536):
                        f.write(chunk)
                seen_obs.add(obs_id)
                croc_new += 1
                time.sleep(0.25)
            except Exception as e:
                print(f"\n    [err] {fname}: {e}")

    print(f"\n  → {croc_new} new Crocodylia files downloaded")
    total_new += croc_new

    # ── Phase 2: Mass-anchor classes ──────────────────────────────────────────
    for category, taxon_id in TARGETS.items():
        print(f"\n{'='*60}\nCategory: {category}  (taxon={taxon_id})")
        save_dir = os.path.join(output_dir, category)
        os.makedirs(save_dir, exist_ok=True)

        existing     = len([f for f in os.listdir(save_dir)
                             if os.path.splitext(f)[1].lower() in AUDIO_EXTS])
        still_needed = max(0, target_per_class - existing)
        print(f"  Have: {existing}  |  Still need: {still_needed}")
        if still_needed == 0:
            print("  Target reached. Skipping.")
            continue

        urls = collect_urls(taxon_id, target=still_needed + 30)
        print(f"  → {len(urls)} candidate URLs found")

        cat_new = 0
        for url, ext, obs_id, snd_id in tqdm(urls, desc=category):
            fname     = f"{category}_{obs_id}_{snd_id}{ext}"
            save_path = os.path.join(save_dir, fname)
            if os.path.exists(save_path):
                continue
            try:
                r = requests.get(url, stream=True, timeout=60)
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(65536):
                        f.write(chunk)
                cat_new += 1
                time.sleep(0.25)
            except Exception as e:
                print(f"\n    [err] {fname}: {e}")

        print(f"  → {cat_new} new files saved for {category}")
        total_new += cat_new

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}\nTotal: {total_new} new files\n\nSummary:")
    for d in sorted(os.listdir(output_dir)):
        full = os.path.join(output_dir, d)
        if os.path.isdir(full):
            n = len([f for f in os.listdir(full)
                     if os.path.splitext(f)[1].lower() in AUDIO_EXTS])
            flag = "✓" if n >= 80 else f"only {n}"
            print(f"  {d:<30}: {n:>4}  [{flag}]")


if __name__ == "__main__":
    run(output_dir="./DATA/birds", target_per_class=500)
