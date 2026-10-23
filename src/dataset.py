"""
Dataset pipeline for Multimodal Fake News Detection.

Key design decisions:
  1. Generates synthetic text-image pairs using procedural PIL images
     and realistic headline corpora across 4 categories.
  2. Pre-extracts CLIP features (text & image) once and caches them
     to disk as .pt tensors, so training is ultra-fast (<1s / epoch).
  3. Negative pairs are created by shuffling image-text associations.
  4. Strict 50:50 class balance is enforced.
  5. 80/20 train/validation split (no test split needed for retraining,
     but a 15% test holdout is kept for final evaluation).

Label convention:
  0 = Consistent / Matched
  1 = Inconsistent / Mismatched
"""

import os
import random
import math
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Suppress HuggingFace Windows symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# ---------------------------------------------------------------------------
# Procedural Image Synthesizers (PIL based) -- unchanged from earlier phases
# ---------------------------------------------------------------------------

def draw_disaster_image(draw: ImageDraw.ImageDraw, width=256, height=256, seed=0):
    rng = random.Random(seed)
    for y in range(height):
        r = int(50 + 160 * (y / height) * 0.8)
        g = int(20 + 40 * math.sin(y / 20.0))
        b = int(10 + 20 * math.cos(y / 30.0))
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    for _ in range(8):
        points = []
        base_x = rng.randint(20, width - 20)
        base_y = height
        top_y = rng.randint(40, height // 2)
        points.append((base_x - rng.randint(20, 50), base_y))
        for y_step in range(base_y - 20, top_y, -25):
            x_jitter = base_x + rng.randint(-30, 30)
            points.append((x_jitter, y_step))
        points.append((base_x, top_y))
        points.append((base_x + rng.randint(20, 50), base_y))
        draw.polygon(points, fill=(rng.randint(220, 255), rng.randint(80, 160), rng.randint(0, 40), 180))
    for i in range(-width, width * 2, 40):
        draw.line([(i, 0), (i + 40, 40)], fill=(240, 200, 20), width=6)
        draw.line([(i, height - 40), (i + 40, height)], fill=(240, 40, 20), width=6)
    for _ in range(35):
        cx = rng.randint(0, width)
        cy = rng.randint(0, height)
        rad = rng.randint(2, 9)
        draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                     fill=(rng.randint(200, 255), rng.randint(140, 220), 50))


def draw_sports_image(draw: ImageDraw.ImageDraw, width=256, height=256, seed=0):
    rng = random.Random(seed)
    for y in range(height):
        stripe = (y // 18) % 2
        g = 135 + (25 if stripe else -15) + int(10 * math.sin(y / 15.0))
        r = 30 + (10 if stripe else 0)
        b = 40 + (15 if stripe else 0)
        draw.line([(0, y), (width, y)], fill=(r, g, b))
    draw.rectangle([20, 20, width - 20, height - 20], outline=(255, 255, 255), width=4)
    draw.line([(20, height // 2), (width - 20, height // 2)], fill=(255, 255, 255), width=3)
    draw.ellipse([width//2 - 40, height//2 - 40, width//2 + 40, height//2 + 40],
                 outline=(255, 255, 255), width=3)
    ball_x, ball_y = rng.randint(70, width - 70), rng.randint(70, height - 70)
    b_rad = 32
    draw.ellipse([ball_x - b_rad, ball_y - b_rad, ball_x + b_rad, ball_y + b_rad],
                 fill=(235, 120, 20), outline=(20, 20, 20), width=3)
    draw.arc([ball_x - b_rad, ball_y - b_rad, ball_x + b_rad, ball_y + b_rad],
             30, 210, fill=(20, 20, 20), width=3)
    for _ in range(15):
        lx, ly = rng.randint(10, width - 10), rng.randint(10, 50)
        draw.rectangle([lx, ly, lx + 8, ly + 6], fill=(255, 240, 80))


def draw_politics_image(draw: ImageDraw.ImageDraw, width=256, height=256, seed=0):
    rng = random.Random(seed)
    half = width // 2
    for x in range(width):
        if x < half - 4:
            c = (20, 35, 95)
        elif x > half + 4:
            c = (145, 25, 35)
        else:
            c = (215, 180, 50)
        draw.line([(x, 0), (x, height)], fill=c)
    for px in range(25, width - 25, 38):
        draw.rectangle([px, 60, px + 16, height - 40], fill=(235, 235, 240), outline=(160, 160, 170), width=2)
        draw.rectangle([px - 4, 52, px + 20, 60], fill=(245, 245, 250))
        draw.rectangle([px - 4, height - 40, px + 20, height - 32], fill=(245, 245, 250))
    draw.polygon([(15, 52), (width // 2, 14), (width - 15, 52)], fill=(240, 240, 245), outline=(140, 140, 150), width=2)
    cx, cy, s_rad = width // 2, height // 2 + 10, 38
    draw.ellipse([cx - s_rad, cy - s_rad, cx + s_rad, cy + s_rad],
                 outline=(230, 195, 50), width=4, fill=(25, 30, 65))
    for i in range(8):
        angle = i * (2 * math.pi / 8)
        sx = cx + int((s_rad - 12) * math.cos(angle))
        sy = cy + int((s_rad - 12) * math.sin(angle))
        draw.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=(255, 230, 80))


def draw_tech_image(draw: ImageDraw.ImageDraw, width=256, height=256, seed=0):
    rng = random.Random(seed)
    for y in range(height):
        b = int(35 + 25 * (y / height))
        draw.line([(0, y), (width, y)], fill=(8, 16, b))
    for x in range(0, width, 20):
        draw.line([(x, 0), (x, height)], fill=(15, 35, 60), width=1)
    for y in range(0, height, 20):
        draw.line([(0, y), (width, y)], fill=(15, 35, 60), width=1)
    cx1, cy1 = width // 2 - 45, height // 2 - 45
    cx2, cy2 = width // 2 + 45, height // 2 + 45
    draw.rectangle([cx1, cy1, cx2, cy2], fill=(25, 28, 38), outline=(0, 235, 215), width=3)
    draw.rectangle([cx1 + 10, cy1 + 10, cx2 - 10, cy2 - 10], fill=(12, 16, 26), outline=(0, 160, 235), width=2)
    for px in range(cx1 + 8, cx2 - 8, 8):
        draw.line([(px, cy1 - 8), (px, cy1)], fill=(220, 180, 50), width=2)
        draw.line([(px, cy2), (px, cy2 + 8)], fill=(220, 180, 50), width=2)
    for py in range(cy1 + 8, cy2 - 8, 8):
        draw.line([(cx1 - 8, py), (cx1, py)], fill=(220, 180, 50), width=2)
        draw.line([(cx2, py), (cx2 + 8, py)], fill=(220, 180, 50), width=2)
    neon_colors = [(0, 240, 255), (0, 255, 170), (160, 80, 255)]
    for _ in range(16):
        col = rng.choice(neon_colors)
        sx = rng.choice([cx1 - 8, cx2 + 8, rng.randint(20, width - 20)])
        sy = rng.choice([cy1 - 8, cy2 + 8, rng.randint(20, height - 20)])
        mx, my = sx + rng.choice([-40, -20, 20, 40]), sy
        ex, ey = mx, my + rng.choice([-40, -20, 20, 40])
        draw.line([(sx, sy), (mx, my), (ex, ey)], fill=col, width=2)
        draw.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=(255, 255, 255))


GENERATOR_MAP = {
    "disasters": draw_disaster_image,
    "sports": draw_sports_image,
    "politics": draw_politics_image,
    "tech": draw_tech_image,
}

# ---------------------------------------------------------------------------
# Headline Corpus
# ---------------------------------------------------------------------------

HEADLINE_CORPUS = {
    "disasters": [
        "Massive Magnitude 7.2 Earthquake Strikes Coastal Province Causing Widespread Structural Damage",
        "Wildfire Rapidly Spreads Across Dry Valley Forcing Emergency Evacuation of 15000 Residents",
        "Category 4 Hurricane Makes Catastrophic Landfall With Sustained 140mph Winds and Severe Surges",
        "Flash Flooding Submerges Major Highways and Downtown Districts After Record Midnight Rainfall",
        "Volcanic Ash Cloud Disrupts International Air Travel as Seismic Activity Surges Dramatically",
        "Emergency Crews Rescue Hundreds Stranded on Rooftops in Worst Regional Inundation in Decades",
        "Historic Blizzard Dumps Four Feet of Snow Paralyzing Metropolitan Transit and Electric Grid",
        "Tsunami Warning Issued Across Pacific Archipelago Following Deep Ocean Undersea Tremor",
        "Industrial Chemical Plant Fire Triggers High-Level Toxicity Alerts and Shelter In Place Order",
        "Severe Drought Depletes Mountain Reservoir Levels to Historic 80-Year Record Lows",
        "Deadly Mudslide Wipes Out Mountain Highway and Threatens Downstream Rural Settlements",
        "Tornado Outbreak Tears Across Midwestern Counties Destroying Homes and Agricultural Silos",
        "Coastal Barrier Island Suffers Extreme Beachfront Erosion Following Consecutive Tropical Storms",
        "Rapid Glacial Lake Outburst Triggers Devastating Downstream Flash Flood in Alpine Valley",
        "Heatwave Breaks All-Time Regional Records Straining Power Grid With Blackouts Reported",
        "Severe Cyclone Leaves Hundreds of Thousands Without Clean Water and Power Across Archipelago",
        "Dam Spillway Exceeds Maximum Capacity Forcing Urgent Midnight Inundation Warnings",
        "Emergency Responders Search Wreckage Following Catastrophic Apartment Complex Collapse",
        "Forest Fire Envelopes National Reserve as Unseasonably Arid Gale Winds Hamper Containment",
        "Heavy Monsoon Downpours Trigger Severe Urban Landslides and Bridge Infrastructure Failures",
    ],
    "sports": [
        "Underdog Franchise Clinches National Championship in Thrilling Double Overtime Victory",
        "Star Quarterback Shatters Decade-Old All-Time Passing Record With Stunning Final Drive",
        "Olympic Sprinter Breaks World 100m Dash Record With Blistering Sub-9.5 Second Time",
        "Championship Final Decided by Dramatic Penalty Shootout Before Sold-Out Stadium Crowd",
        "Veteran Tennis Champion Triumphs in Five-Set Grand Slam Epic on Historic Center Court",
        "Home Team Pulls Off Unbelievable 20-Point Comeback in Fourth Quarter of Title Clash",
        "Cycling Tour Contender Dominates Grueling Mountain Stage to Claim Iconic Yellow Jersey",
        "Marathon Runner Smashes Course Milestone by Over Two Minutes in Rainy Capital Race",
        "Rising Tennis Phenom Upsets Top-Ranked Seed in Sensational Straight Sets Masterclass",
        "Defending Champions Hoist Gold Trophy After Undefeated Regular Season Performance",
        "Basketball MVP Explodes for 55 Points in Decisive Playoff Series Clinching Match",
        "Goalkeeper Makes Four Spectacular Stoppages in Stoppage Time to Seal League Championship",
        "Heavyweight Boxing Championship Belt Decided by Stunning Ninth Round Knockout",
        "Formula Motorsport Driver Takes Flag-to-Flag Victory in Wet Weather Grand Prix Classic",
        "University Athletics Team Secures Historic Triple Crown in National Track and Field",
        "National Soccer Squad Qualifies for Global Tournament With Stoppage Time Header",
        "Baseball Ace Tosses Historic No-Hitter Striking Out Fourteen in Masterful Display",
        "Rowing Crew Powers Through Headwind to Capture International Regatta Gold by Split Second",
        "Gymnast Delivers Flawless Floor Routine to Earn Perfect Score and All-Around Gold",
        "Ice Hockey Goaltender Records 45 Saves in Shutout Victory to Hoist Coveted Cup",
    ],
    "politics": [
        "Parliament Passes Historic Bipartisan Environmental Protection Bill Following Heated Debate",
        "Head of State Delivers Annual Address Pledging Ambitious Infrastructure and Economic Reforms",
        "High Court Delivers Landmark Constitutional Verdict Overhauling Campaign Finance Regulations",
        "Diplomatic Leaders Sign Historic International Peace Accord at Geneva Summit",
        "National Election Turnout Surges to Record Highs as Citizens Vote Across Fifty Districts",
        "Treasury Ministry Unveils Comprehensive Federal Budget Targeting Inflation and Healthcare",
        "Congressional Committee Concludes Month-Long Public Hearings on National Security Policy",
        "Opposition Coalition Wins Crucial Parliamentary Majority in Decisive General Election",
        "Bilateral Trade Delegation Reaches Milestone Treaty Reducing Cross-Border Tariffs",
        "Cabinet Approves Nationwide Energy Modernization Plan With Broad Legislative Backing",
        "Special Senate Committee Convenes Emergency Inquiry Regarding Financial Regulatory Oversight",
        "Prime Minister Announces Comprehensive Cabinet Reshuffle Ahead of Legislative Session",
        "Regional Governors Sign Unified Interstate Compact on Public Healthcare Coordination",
        "Electoral Commission Confirms Final Certified Results After Comprehensive Recount",
        "Diplomatic Envoy Dispatched to Facilitate Bilateral Ceasefire Negotiations",
        "Supreme Judicial Council Appoints Three New Justices in Landmark Confirmation Ceremony",
        "Civic Leaders Gather for Tri-City Mayoral Summit on Urban Housing and Infrastructure",
        "Federal Agency Implements Stringent Transparency Standards for Government Procurement",
        "Legislators Reach Compromise Agreement on Comprehensive National Defense Authorization",
        "Global Leaders Unveil Joint Declaration on Sovereign Cybersecurity Cooperation",
    ],
    "tech": [
        "Tech Giant Unveils Revolutionary Next-Generation Quantum Computing Processor Architecture",
        "Artificial Intelligence Startup Achieves Human-Level Multimodal Reasoning Benchmark",
        "Semiconductor Foundry Begins Commercial Production of Sub-2nm Silicon Microchips",
        "Breakthrough Solid-State Battery Prototype Quadruples Electric Vehicle Travel Range",
        "Aerospace Engineers Successfully Complete Autonomous Suborbital Flight Test",
        "Cybersecurity Researchers Discover and Patch Critical Zero-Day Vulnerability in Web Infrastructure",
        "Commercial Fusion Energy Reactor Sustains Net Energy Positive Plasma Reaction for Record Duration",
        "Novel Neural Interface Allows Direct Mind-to-Text Transcription at 90 Words Per Minute",
        "Open-Source Foundation Releases Ultra-Fast 100-Billion Parameter Foundation Model",
        "Robotics Consortium Demonstrates Fully Bipedal Humanoid Completing Complex Industrial Tasks",
        "Optical Computing Prototype Transmits Petabits of Data Using Silicon Photonic Waveguides",
        "Next-Generation Operating System Kernel Rewritten in Rust for Formally Verified Memory Safety",
        "Autonomous Electric Vertical Takeoff Aircraft Completes First Multi-Passenger Flight",
        "Space Exploration Agency Deploys Next-Gen Deep Space Imaging Telescope Array",
        "Researchers Develop Room-Temperature Superconducting Composite With Repeatable Verification",
        "Cloud Computing Provider Launches World's Largest Exascale AI Supercomputer Cluster",
        "Autonomous Fleet Achieves 50 Million Driverless Miles With Zero High-Severity Incidents",
        "CRISPR-Cas9 Gene Editing Platform Receives Historic Regulatory Approval for Rare Diseases",
        "Micro-LED Display Technology Eliminates Pixel Burn-In While Halving Power Consumption",
        "Next-Gen 6G Wireless Laboratory Demonstrates Terabit Wireless Transmission Through Obstacles",
    ],
}

# ---------------------------------------------------------------------------
# Synthetic Benchmark Generator
# ---------------------------------------------------------------------------

def generate_synthetic_benchmark(
    data_dir: str = "data",
    num_per_category: int = 40,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Creates a balanced synthetic multimodal benchmark.

    For each category:
      - `num_per_category` matched pairs (label 0 = Consistent).
      - `num_per_category` mis-matched pairs (label 1 = Inconsistent),
        created by pairing the headline with an image from a *different* category.

    Total = 4 categories x 2 x num_per_category = 320 pairs (default).
    Enforces strict 50:50 class balance.
    Split: 70% train, 15% val, 15% test.
    """
    rng = random.Random(random_seed)
    np.random.seed(random_seed)

    data_path = Path(data_dir)
    images_dir = data_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    categories = ["disasters", "sports", "politics", "tech"]
    records: List[Dict] = []
    sample_id = 0

    print(f"[Dataset] Generating balanced benchmark ({num_per_category}x2x{len(categories)} = {num_per_category*2*len(categories)} pairs)...")

    for cat in categories:
        headlines = HEADLINE_CORPUS[cat]
        mismatch_cats = [c for c in categories if c != cat]

        # --- Consistent / Matched Pairs (Label 0) ---
        for i in range(num_per_category):
            sample_id += 1
            headline = headlines[i % len(headlines)]
            fname = f"img_{sample_id:04d}.png"
            img_path = images_dir / fname

            img = Image.new("RGB", (256, 256), (0, 0, 0))
            d = ImageDraw.Draw(img, "RGBA")
            GENERATOR_MAP[cat](d, 256, 256, seed=sample_id * 17)
            img.save(img_path, format="PNG")

            records.append({
                "id": sample_id,
                "category": cat,
                "headline": headline,
                "image_category": cat,
                "image_path": str(Path("data") / "images" / fname),
                "label": 0,
                "label_name": "Consistent",
            })

        # --- Inconsistent / Mismatched Pairs (Label 1) ---
        for i in range(num_per_category):
            sample_id += 1
            headline = headlines[i % len(headlines)]
            mis_cat = mismatch_cats[i % len(mismatch_cats)]
            fname = f"img_{sample_id:04d}.png"
            img_path = images_dir / fname

            img = Image.new("RGB", (256, 256), (0, 0, 0))
            d = ImageDraw.Draw(img, "RGBA")
            GENERATOR_MAP[mis_cat](d, 256, 256, seed=sample_id * 31)
            img.save(img_path, format="PNG")

            records.append({
                "id": sample_id,
                "category": cat,
                "headline": headline,
                "image_category": mis_cat,
                "image_path": str(Path("data") / "images" / fname),
                "label": 1,
                "label_name": "Inconsistent",
            })

    df = pd.DataFrame(records)

    # Stratified Split: Train 70% / Val 15% / Test 15%
    df["strat_key"] = df["category"] + "_" + df["label"].astype(str)
    df["split"] = ""
    for _, grp in df.groupby("strat_key"):
        idxs = grp.index.tolist()
        rng.shuffle(idxs)
        n = len(idxs)
        n_train = int(round(n * 0.70))
        n_val = int(round(n * 0.15))
        df.loc[idxs[:n_train], "split"] = "train"
        df.loc[idxs[n_train:n_train + n_val], "split"] = "val"
        df.loc[idxs[n_train + n_val:], "split"] = "test"
    df.drop(columns=["strat_key"], inplace=True)

    csv_path = data_path / "dataset.csv"
    df.to_csv(csv_path, index=False)
    print(f"[Dataset] Saved {len(df)} pairs -> {csv_path}")
    print(f"  Label distribution: {dict(df['label'].value_counts())}")
    print(f"  Split distribution: {dict(df['split'].value_counts())}")

    # Also save a test political portrait for UI demos
    portrait_path = images_dir / "test_political_portrait.png"
    pimg = Image.new("RGB", (256, 256), (0, 0, 0))
    pd_draw = ImageDraw.Draw(pimg, "RGBA")
    draw_politics_image(pd_draw, 256, 256, seed=999)
    pimg.save(portrait_path, format="PNG")

    return df


# ---------------------------------------------------------------------------
# Pre-extract CLIP features to .pt cache files
# ---------------------------------------------------------------------------

def preextract_clip_features(
    csv_path: str = "data/dataset.csv",
    cache_dir: str = "data/clip_cache",
    model_name: str = "openai/clip-vit-base-patch32",
    batch_size: int = 32,
):
    """
    Pre-extracts CLIP text and image features for every sample,
    saves them as a single .pt file.  Subsequent training loads these
    tensors directly, skipping the CLIP forward pass entirely.
    """
    from transformers import CLIPModel, CLIPProcessor

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path / "features.pt"

    if cache_file.exists():
        print(f"[Cache] Loading pre-extracted features from {cache_file}")
        return torch.load(cache_file, weights_only=False)

    df = pd.read_csv(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Cache] Extracting CLIP features on {device} for {len(df)} samples...")

    model = CLIPModel.from_pretrained(model_name, use_safetensors=True).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    all_text_feats = []
    all_img_feats = []
    all_labels = []
    all_splits = []

    for start in range(0, len(df), batch_size):
        batch_df = df.iloc[start: start + batch_size]
        texts = batch_df["headline"].tolist()
        labels = batch_df["label"].tolist()
        splits = batch_df["split"].tolist()

        # Load images
        images = []
        for img_p in batch_df["image_path"]:
            images.append(Image.open(img_p).convert("RGB"))

        with torch.no_grad():
            text_inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
            img_inputs = processor(images=images, return_tensors="pt").to(device)

            text_feats = model.get_text_features(**text_inputs)
            img_feats = model.get_image_features(**img_inputs)

            # L2 normalise
            text_feats = F.normalize(text_feats, p=2, dim=-1)
            img_feats = F.normalize(img_feats, p=2, dim=-1)

        all_text_feats.append(text_feats.cpu())
        all_img_feats.append(img_feats.cpu())
        all_labels.extend(labels)
        all_splits.extend(splits)

        if (start // batch_size) % 5 == 0:
            print(f"  ... processed {start + len(batch_df)}/{len(df)}")

    cache_data = {
        "text_features": torch.cat(all_text_feats, dim=0),   # (N, 512)
        "image_features": torch.cat(all_img_feats, dim=0),   # (N, 512)
        "labels": torch.tensor(all_labels, dtype=torch.float32),  # (N,)
        "splits": all_splits,
    }
    torch.save(cache_data, cache_file)
    print(f"[Cache] Saved pre-extracted features -> {cache_file}")
    print(f"  text_features  : {cache_data['text_features'].shape}")
    print(f"  image_features : {cache_data['image_features'].shape}")
    print(f"  labels         : {cache_data['labels'].shape}")

    # Cleanup GPU memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return cache_data


# ---------------------------------------------------------------------------
# PyTorch Dataset operating on pre-cached CLIP features
# ---------------------------------------------------------------------------

class CLIPFeatureDataset(Dataset):
    """
    Loads pre-extracted, L2-normalised CLIP features from cache.
    Each item: (text_features, image_features, label).
    """

    def __init__(self, cache_data: Dict, split: Optional[str] = None):
        splits = cache_data["splits"]
        if split is not None:
            mask = [s == split for s in splits]
            indices = [i for i, m in enumerate(mask) if m]
        else:
            indices = list(range(len(splits)))

        self.text_features = cache_data["text_features"][indices]
        self.image_features = cache_data["image_features"][indices]
        self.labels = cache_data["labels"][indices]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "text_features": self.text_features[idx],
            "image_features": self.image_features[idx],
            "label": self.labels[idx],
        }


def get_feature_dataloaders(
    cache_data: Dict,
    batch_size: int = 32,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Returns train, val, test DataLoaders from pre-cached features."""
    train_ds = CLIPFeatureDataset(cache_data, split="train")
    val_ds = CLIPFeatureDataset(cache_data, split="val")
    test_ds = CLIPFeatureDataset(cache_data, split="test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Verification entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 65)
    print("DATASET PIPELINE & CLIP FEATURE EXTRACTION")
    print("=" * 65)

    # 1. Generate synthetic benchmark
    df = generate_synthetic_benchmark(data_dir="data", num_per_category=40, random_seed=42)

    print(f"\n--- DATASET SUMMARY ---")
    print(f"Total Samples : {len(df)}")
    print(f"Categories    : {list(df['category'].unique())}")
    print(f"Labels (0=Match, 1=Mismatch):")
    print(df["label"].value_counts().to_string())
    print(f"\nSplits:")
    print(df["split"].value_counts().to_string())

    # 2. Pre-extract CLIP features
    cache = preextract_clip_features(csv_path="data/dataset.csv")

    # 3. Verify dataloaders
    train_loader, val_loader, test_loader = get_feature_dataloaders(cache, batch_size=32)
    print(f"\nDataLoader sizes: Train={len(train_loader.dataset)} Val={len(val_loader.dataset)} Test={len(test_loader.dataset)}")

    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  text_features  : {tuple(batch['text_features'].shape)}")
    print(f"  image_features : {tuple(batch['image_features'].shape)}")
    print(f"  label          : {tuple(batch['label'].shape)}")
    print(f"  label values   : {batch['label'].tolist()[:10]}...")

    # Quick sanity: cosine similarity distribution
    cos = torch.sum(batch['text_features'] * batch['image_features'], dim=-1)
    labels = batch['label']
    match_cos = cos[labels == 0].mean().item() if (labels == 0).any() else 0
    mismatch_cos = cos[labels == 1].mean().item() if (labels == 1).any() else 0
    print(f"\n  Mean cos-sim (matched)    : {match_cos:.4f}")
    print(f"  Mean cos-sim (mismatched) : {mismatch_cos:.4f}")
    print(f"  Separation gap            : {match_cos - mismatch_cos:.4f}")

    print("\n" + "=" * 65)
    print("DATASET PIPELINE VERIFIED OK")
    print("=" * 65)
