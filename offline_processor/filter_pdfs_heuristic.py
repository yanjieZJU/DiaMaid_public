"""
Find articles with artifacts
"""

import os
import re
import csv
import sys
import argparse
import fitz  # PyMuPDF
from tqdm import tqdm

# ================= Configuration =================

# User defined inputs
# If running locally, you might need to adjust this path or pass it as an argument
DEFAULT_SEARCH_PATH = r"E:\article_informations\original_articles"
OUTPUT_CSV_PATH = "pdf_scan_report.csv"
THUMBNAIL_DIR = "./thumbnails"

# Keywords found in Figure captions that indicate architecture
TARGET_KEYWORDS = [
    "architecture", "framework", "overview", "pipeline", "system", 
    "workflow", "model structure", "process"
]

# Conferences/Journals to prioritize (folder name matching)
PRIORITY_FOLDERS = ["dis"]

# ================= Functions =================

def ensure_fitz_installed():
    try:
        import fitz
        return True
    except ImportError:
        print("Error: PyMuPDF (fitz) is not installed.")
        print("Please run: pip install pymupdf")
        return False

def is_priority_folder(folder_path):
    """Check if the folder path contains priority conference names."""
    norm_path = folder_path.lower()
    for key in PRIORITY_FOLDERS:
        if key.lower() in norm_path:
            return True
    return False

def clean_text(text):
    """Normalize whitespace."""
    return re.sub(r'\s+', ' ', text).strip()

def extract_captions_and_images(pdf_path, thumbnail_dir, rel_path_id):
    """
    Scans PDF (first 5 pages) for:
    1. Figure captions containing keywords.
    2. Large images to save as thumbnails.
    """
    results = {
        "has_keywords": False,
        "matched_caption": "",
        "thumbnail_path": "",
        "page_num": -1
    }
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"Error opening {pdf_path}: {e}")
        return results

    # Limit to first 5 pages + last page (sometimes architecture is at the end? usually beginning)
    # Most system architectures are in pages 1-4.
    pages_to_scan = range(min(5, len(doc)))
    
    found_caption = False
    
    for page_num in pages_to_scan:
        page = doc[page_num]
        text = page.get_text("text")
        
        # --- 1. Caption Search ---
        # Look for "Figure X: ... keyword ..."
        # OR "Fig. X ... keyword ..."
        # Regex explanation:
        # (?i) : Case insensitive
        # (?:Figure|Fig\.?) : Match "Figure" or "Fig" or "Fig."
        # \s*[\d\.]+ : Match number like "1", "2.1"
        # [^:\n]*[:\.]? : Match " 1" or ": " or "." optionally
        # .*? : Match content (non-greedy)
        # ({keywords}) : Match one of the keywords
        keyword_pattern = "|".join(TARGET_KEYWORDS)
        pattern = fr"(?i)(?:Figure|Fig\.?)\s*[\d\.]+[^\n]*?({keyword_pattern})"
        
        matches = re.finditer(pattern, text)
        for match in matches:
            # Get a window of text around the match to ensure it's a real caption context
            # We treat the line or checking approx 200 chars as the caption
            start = max(0, match.start())
            end = min(len(text), match.end() + 150) # Catch context
            snippet = clean_text(text[start:end])
            
            if not found_caption:
                results["has_keywords"] = True
                results["matched_caption"] = snippet
                results["page_num"] = page_num + 1
                found_caption = True # Mark found, but continue to look for images
        
        # --- 2. Image Extraction (Visual Aid) ---
        # We try to extract the largest image on this page if we found a caption, 
        # OR we just extract the largest image seen so far in first 5 pages 
        # that looks like a diagram (w>300, h>200).
        
        # Get images on page
        image_list = page.get_images(full=True)
        
        best_image_area = 0
        
        if image_list:
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                ext = base_image["ext"]
                
                # Check dimensions logic via pixmap for fast check without full decode? 
                # extract_image gives width/height
                w = base_image["width"]
                h = base_image["height"]
                
                # Filter small icons
                if w > 300 and h > 200:
                    area = w * h
                    # Save if it's the largest structure-like image we've seen OR it matches the caption page
                    # Heuristic: If we found a caption on this page, this might be THE image.
                    # If we haven't found a caption yet, keep the largest candidate.
                    
                    is_on_caption_page = (page_num + 1 == results["page_num"])
                    
                    # Logic: Always update if on caption page (preferentially)
                    # Otherwise verify against previous best
                    
                    # We usually want one thumbnail per paper.
                    # If we already have a thumbnail from a "caption page", skip others.
                    # Unless this is also on the caption page and bigger.
                    
                    thumb_name = f"{rel_path_id}_p{page_num+1}_{img_index}.{ext}"
                    thumb_full_path = os.path.join(thumbnail_dir, thumb_name)
                    
                    # Write file
                    # To avoid saving too many, we only save temporarily or overwrite?
                    # Let's save the largest one encountered overall, but prioritize the one on the caption page.
                    
                    # Actually, for simplicity in this script, let's just save the FIRST image found on a page with a relevant caption.
                    # If no caption found yet, we skip saving for now to save space, OR save largest from pg 1-2.
                    
                    should_save = False
                    if results["has_keywords"] and is_on_caption_page:
                        should_save = True
                    elif not results["has_keywords"] and page_num < 2:
                        # Fallback: Save largest from first 2 pages if no keywords found yet
                        # (But we might overwrite later if keyword found)
                        should_save = True
                    
                    if should_save:
                        # Save
                        with open(thumb_full_path, "wb") as f:
                            f.write(image_bytes)
                        results["thumbnail_path"] = thumb_full_path
                        # If we found a caption match on this page and saved an image, we are likely done for this paper
                        if results["has_keywords"]:
                            break 

        if results["has_keywords"] and results["thumbnail_path"]:
            break # Found both caption and image, stop scanning this PDF

    doc.close()
    return results

def main():
    if not ensure_fitz_installed():
        return

    parser = argparse.ArgumentParser(description="Scan PDFs for architecture diagrams.")
    parser.add_argument("--path", default=DEFAULT_SEARCH_PATH, help="Root directory to scan")
    parser.add_argument("--priority", action="store_true", help="Only scan priority folders (UIST, CHI, etc.)")
    args = parser.parse_args()

    input_root = args.path
    if not os.path.exists(input_root):
        print(f"Directory not found: {input_root}")
        return

    # Create thumbnail dir
    os.makedirs(THUMBNAIL_DIR, exist_ok=True)
    
    # Prepare CSV
    headers = ["Filename", "Conference/Folder", "Found_Keywords", "Matched_Caption", "Page", "Thumbnail", "FullPath"]
    
    rows = []
    
    print(f"Scanning {input_root}...")
    if args.priority:
        print(f"Filtering for folders containing: {PRIORITY_FOLDERS}")
    
    # Walk directory
    pdf_files = []
    for root, dirs, files in os.walk(input_root):
        folder_name = os.path.basename(root)
        
        if args.priority:
            if not is_priority_folder(root):
                continue
        
        for f in files:
            if f.lower().endswith(".pdf"):
                pdf_files.append((os.path.join(root, f), folder_name, f))

    print(f"Found {len(pdf_files)} PDF files. Processing...")

    # Process
    with open(OUTPUT_CSV_PATH, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        
        for full_path, folder, filename in tqdm(pdf_files):
            # Unique ID for thumbnail (cleaned filename)
            safe_name = "".join(c for c in filename if c.isalnum() or c in ('-','_'))[:30]
            rel_id = f"{folder}_{safe_name}"
            
            data = extract_captions_and_images(full_path, THUMBNAIL_DIR, rel_id)
            
            # Decide inclusion:
            # Option A: Include ALL papers in the list
            # Option B: Include ONLY matches
            # User said "Screening... from a folder", implies they want to see the hits.
            # But "traversal filtering" implies we might want to see what was rejected?
            # Let's write ALL rows but sort/mark them. 
            # Or better, just write matches if the goal is "screening out".
            # "Target is to screen OUT [extracted]... articles"
            
            # Let's write all but emphasize matches. 
            row = [
                filename,
                folder,
                "YES" if data["has_keywords"] else "NO",
                data["matched_caption"],
                data["page_num"] if data["page_num"] != -1 else "",
                data["thumbnail_path"],
                full_path
            ]
            
            writer.writerow(row)
            csvfile.flush() # Ensure wrote

    print(f"\nCompleted. Results saved to {OUTPUT_CSV_PATH}")
    print(f"Thumbnails saved to {THUMBNAIL_DIR}")

if __name__ == "__main__":
    main()
