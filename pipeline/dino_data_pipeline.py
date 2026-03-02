import os
import glob
import numpy as np
import torch
import torchaudio.transforms as T
from pydub import AudioSegment
from tqdm import tqdm

# Ensure pydub finds the conda-installed ffmpeg
AudioSegment.converter = "/home/rahulraj/miniconda3/bin/ffmpeg"


class DinoDataPipeline:
    """
    Converts a directory of raw .mp3 audio files into normalized
    Log-Mel Spectrogram .pt tensors ready for the Dino-VAE.

    Each audio file is sliced into non-overlapping 3-second chunks.
    Output shape per tensor: (1, 128, 259)  [channels x mel_bins x time_frames]
    """
    def __init__(self, input_dir="./DATA/birds", output_dir="./DATA/tensors"):
        self.input_dir  = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Audio standards
        self.target_sr   = 22050
        self.clip_length = self.target_sr * 3   # 3-second chunks

        # Mel-Spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.target_sr,
            n_fft=1024,
            hop_length=256,
            n_mels=128,
        )
        self.amplitude_to_db = T.AmplitudeToDB()

    def _process_waveform(self, waveform, sr):
        """Resample → mono."""
        if sr != self.target_sr:
            waveform = T.Resample(sr, self.target_sr)(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        return waveform

    def _to_spectrogram(self, chunk):
        """Convert a (1, N) waveform chunk to a normalized log-mel spectrogram."""
        mel     = self.mel_transform(chunk)
        log_mel = self.amplitude_to_db(mel)
        denom   = log_mel.max() - log_mel.min()
        if denom < 1e-6:   # silent chunk — skip
            return None
        return (log_mel - log_mel.min()) / denom

    def _load_audio(self, path):
        """Load any audio format via pydub/ffmpeg → (waveform, sample_rate)."""
        audio   = AudioSegment.from_file(path)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        max_val = float(2 ** (8 * audio.sample_width - 1))
        samples = samples / max_val  # normalize to [-1, 1]

        if audio.channels == 2:
            samples = samples.reshape((-1, 2)).T  # (2, N)
        else:
            samples = samples[np.newaxis, :]       # (1, N)

        return torch.from_numpy(samples), audio.frame_rate

    def process_file(self, mp3_path, species, file_idx):
        """Slice one audio file into 3-second tensor chunks and save them."""
        try:
            waveform, sr = self._load_audio(mp3_path)
        except Exception as e:
            print(f"  [err] Could not load {mp3_path}: {e}")
            return 0

        waveform  = self._process_waveform(waveform, sr)
        total_len = waveform.shape[1]
        n_chunks  = total_len // self.clip_length

        if n_chunks == 0:
            # File shorter than 3 s — pad and save as one chunk
            pad      = self.clip_length - total_len
            waveform = torch.nn.functional.pad(waveform, (0, pad))
            n_chunks = 1

        save_dir = os.path.join(self.output_dir, species)
        os.makedirs(save_dir, exist_ok=True)

        saved = 0
        for i in range(n_chunks):
            chunk = waveform[:, i * self.clip_length:(i + 1) * self.clip_length]
            spec  = self._to_spectrogram(chunk)
            if spec is None:
                continue
            out_path = os.path.join(save_dir, f"{species}_{file_idx:04d}_chunk{i:03d}.pt")
            torch.save(spec, out_path)
            saved += 1

        return saved

    def run(self):
        """Walk input_dir, find all .mp3 files grouped by species subfolder."""
        species_dirs = sorted([
            d for d in os.listdir(self.input_dir)
            if os.path.isdir(os.path.join(self.input_dir, d))
        ])

        if not species_dirs:
            print(f"No species subdirectories found in {self.input_dir}. "
                  "Run xeno_canto_downloader.py first.")
            return

        grand_total = 0
        for species in species_dirs:
            mp3s = sorted(glob.glob(os.path.join(self.input_dir, species, "*.mp3")))
            print(f"\n[{species}] Processing {len(mp3s)} files...")

            species_total = 0
            for idx, mp3_path in enumerate(tqdm(mp3s, desc=species, unit="file")):
                species_total += self.process_file(mp3_path, species, idx)

            print(f"  → {species_total} tensors saved for {species}")
            grand_total += species_total

        print(f"\n{'='*50}")
        print(f"Preprocessing complete. {grand_total} total .pt tensors saved to {self.output_dir}")

        print("\nTensor counts per class:")
        for species in species_dirs:
            pts = glob.glob(os.path.join(self.output_dir, species, "*.pt"))
            print(f"  {species:<15}: {len(pts)}")


if __name__ == "__main__":
    pipeline = DinoDataPipeline(
        input_dir="./DATA/birds",
        output_dir="./DATA/tensors",
    )
    pipeline.run()
