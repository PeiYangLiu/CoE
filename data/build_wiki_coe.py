"""
Wiki-CoE Dataset Construction Pipeline

Builds the Wiki-CoE dataset from 2WikiMultiHopQA by:
1. Crawling Wikipedia entity pages as screenshots via Selenium
2. Annotating bounding boxes for supporting facts via HTML element matching
3. Quality assurance filtering
"""

import os
import re
import json
import time
import hashlib
import logging
import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException, WebDriverException, NoSuchElementException
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Entity priority ranking
# ---------------------------------------------------------------------------

def rank_entities_by_frequency(dataset: List[dict]) -> List[Tuple[str, int]]:
    """Rank entities by how many distinct questions reference them as evidence."""
    entity_counter = Counter()
    for sample in dataset:
        supporting_facts = sample.get("supporting_facts", [])
        # 2WikiMultiHopQA format: [[title, sent_id], ...]
        if isinstance(supporting_facts, list):
            unique_titles = set(pair[0] for pair in supporting_facts if isinstance(pair, list))
        elif isinstance(supporting_facts, dict):
            unique_titles = set(supporting_facts.get("title", []))
        else:
            unique_titles = set()
        for title in unique_titles:
            entity_counter[title] += 1
    return entity_counter.most_common()


# ---------------------------------------------------------------------------
# 2. Offline bbox computation from saved layout
# ---------------------------------------------------------------------------

_CSS_NOISE_RE = re.compile(r"\.mw-parser-output[^{]*\{[^}]*\}", re.DOTALL)
_AT_RULE_RE = re.compile(r"@media[^{]*\{[^{}]*(?:\{[^}]*\}[^{}]*)*\}", re.DOTALL)


def _clean_text(text: str) -> str:
    """Remove embedded CSS style blocks that leak into MediaWiki element text."""
    text = _CSS_NOISE_RE.sub(" ", text)
    text = _AT_RULE_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _compute_bboxes_from_layout(layout: dict) -> List[dict]:
    """
    Compute paragraph bounding boxes from a saved layout snapshot.
    This runs offline (no browser needed) and can be re-run anytime
    with different logic without re-crawling.
    
    Strategy:
    1. Detect float/infobox elements to find the right boundary to clip
    2. For each text element (p, td, li, etc.), use saved line_rects
       to compute tight text-only bbox, clipped to exclude float area
    """
    elements = layout.get("elements", [])
    page_width = layout.get("page_width", 9999)

    # Step 1: collect all float/infobox regions as (x_left, y_top, y_bottom) tuples.
    # A text line's right edge is only clipped to a float's left edge if the
    # line's y-range overlaps the float's y-range. This avoids over-clipping
    # paragraphs that appear BELOW an infobox.
    float_regions: List[Tuple[int, int, int]] = []
    for el in elements:
        cls = el.get("cls", "")
        css_float = el.get("float", "none")
        bcr = el.get("bcr", [0, 0, 0, 0])
        fx1, fy1, fx2, fy2 = bcr
        fw = fx2 - fx1
        fh = fy2 - fy1

        is_float = (
            css_float == "right"
            or "infobox" in cls.lower()
            or "vcard" in cls.lower()
            or "sidebar" in cls.lower()
            or "floatright" in cls.lower()
            or ("thumb" in cls.lower() and "tright" in cls.lower())
        )
        if is_float and fw > 50 and fh > 20:
            float_regions.append((int(fx1) - 10, int(fy1), int(fy2)))

    def _clip_x2(x2: int, y1: int, y2: int) -> int:
        """Clip x2 only if the line vertically overlaps some float region."""
        for fx_left, fy_top, fy_bot in float_regions:
            if y2 > fy_top and y1 < fy_bot:
                if x2 > fx_left:
                    x2 = fx_left
        return x2

    # Step 2: Extract text bboxes with vertically-aware float clipping
    text_tags = {"P", "TD", "TH", "LI", "DD", "DT", "CAPTION", "FIGCAPTION"}
    results = []

    for el in elements:
        if el.get("tag") not in text_tags:
            continue
        text = _clean_text(el.get("text", ""))
        if len(text) == 0:
            continue

        # Use line_rects if available (from Range API), otherwise fall back to bcr
        line_rects = el.get("line_rects", [])
        if line_rects:
            min_x, min_y = float("inf"), float("inf")
            max_x, max_y = float("-inf"), float("-inf")
            for lr in line_rects:
                x1, y1, x2, y2 = lr
                x2 = _clip_x2(int(x2), int(y1), int(y2))
                if x2 <= x1:
                    continue
                min_x = min(min_x, x1)
                min_y = min(min_y, y1)
                max_x = max(max_x, x2)
                max_y = max(max_y, y2)

            if min_x < float("inf"):
                w = max_x - min_x
                h = max_y - min_y
                if w > 30 and h > 5 and h < 800:
                    results.append({
                        "text": text[:2000],
                        "bbox": [int(min_x), int(min_y), int(max_x), int(max_y)],
                    })
        else:
            # Fallback: use bcr with per-line-style vertical clipping
            bcr = el.get("bcr", [0, 0, 0, 0])
            x1, y1, x2, y2 = bcr
            x2 = _clip_x2(int(x2), int(y1), int(y2))
            w, h = x2 - x1, y2 - y1
            if w > 30 and h > 5 and h < 800:
                results.append({
                    "text": text[:2000],
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

    return results


def recompute_bboxes_from_layouts(screenshot_dir: str):
    """
    Re-generate all _bboxes.json files from saved _layout.json files.
    Call this after changing _compute_bboxes_from_layout() logic.
    No browser/crawling needed.
    """
    import glob
    layout_files = glob.glob(os.path.join(screenshot_dir, "*_layout.json"))
    logger.info(f"Recomputing bboxes for {len(layout_files)} entities from saved layouts...")
    for lf in tqdm(layout_files, desc="Recomputing bboxes"):
        with open(lf, "r") as f:
            layout = json.load(f)
        bboxes = _compute_bboxes_from_layout(layout)
        bf = lf.replace("_layout.json", "_bboxes.json")
        with open(bf, "w") as f:
            json.dump(bboxes, f)
    logger.info("Done.")


# ---------------------------------------------------------------------------
# 3. Selenium screenshot crawler
# ---------------------------------------------------------------------------

class WikiScreenshotCrawler:
    """Capture high-resolution Wikipedia page screenshots with Selenium."""

    VIEWPORT_WIDTH = 1280
    MAX_PAGE_HEIGHT = 15000  # cap to avoid extremely long pages

    def __init__(self, output_dir: str, headless: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.driver = self._create_driver(headless)

    def _create_driver(self, headless: bool) -> webdriver.Chrome:
        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument(f"--window-size={self.VIEWPORT_WIDTH},900")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-software-rasterizer")
        options.add_argument("--disable-setuid-sandbox")
        # Use a unique debugging port per instance to allow parallel browsers
        import tempfile, random
        debug_port = random.randint(10000, 60000)
        options.add_argument(f"--remote-debugging-port={debug_port}")
        # Custom user-data-dir to avoid snap sandbox issues
        self._tmpdir = tempfile.mkdtemp(prefix="coe_chrome_")
        options.add_argument(f"--user-data-dir={self._tmpdir}")
        # Use standard user-agent
        options.add_argument(
            "--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        # Try chromium-browser path
        for binary in ["/usr/bin/chromium-browser", "/usr/bin/chromium", "/usr/bin/google-chrome"]:
            if os.path.exists(binary):
                options.binary_location = binary
                break
        driver = webdriver.Chrome(options=options)
        return driver

    def _entity_to_url(self, entity_name: str) -> str:
        slug = entity_name.replace(" ", "_")
        return f"https://en.wikipedia.org/wiki/{slug}"

    def _safe_filename(self, entity_name: str) -> str:
        h = hashlib.md5(entity_name.encode()).hexdigest()[:8]
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in entity_name)
        return f"{safe[:80]}_{h}"

    def capture(self, entity_name: str, retries: int = 3) -> Optional[dict]:
        """Capture a Wikipedia page screenshot. Returns metadata dict or None."""
        url = self._entity_to_url(entity_name)
        fname = self._safe_filename(entity_name)
        img_path = self.output_dir / f"{fname}.png"
        html_path = self.output_dir / f"{fname}.html"

        if img_path.exists() and html_path.exists():
            # Check if layout snapshot exists; if yes, skip (can recompute bboxes offline)
            layout_path = self.output_dir / f"{fname}_layout.json"
            if layout_path.exists():
                return self._load_existing_metadata(entity_name, img_path, html_path)
            # If only old bboxes exist (no layout), also skip — layout is optional
            bboxes_path = self.output_dir / f"{fname}_bboxes.json"
            if bboxes_path.exists():
                return self._load_existing_metadata(entity_name, img_path, html_path)
            # Otherwise fall through to re-capture

        for attempt in range(retries):
            try:
                self.driver.get(url)
                # Wait for main content to load
                WebDriverWait(self.driver, 15).until(
                    EC.presence_of_element_located((By.ID, "bodyContent"))
                )
                time.sleep(1)  # allow images/CSS to render

                # Remove banners and popups that interfere with screenshots
                self.driver.execute_script(
                    'var selectors = ".mw-indicators, .sistersitebox, .navbox, '
                    '.ambox, .mbox-small, .noprint, #mw-panel, #mw-head, '
                    '#mw-navigation, .vector-header, .vector-column-start, '
                    '#footer, .mw-footer";'
                    'document.querySelectorAll(selectors).forEach('
                    'function(e) { e.remove(); });'
                )

                # Get full page dimensions
                page_height = self.driver.execute_script(
                    "return document.documentElement.scrollHeight"
                )
                page_height = min(page_height, self.MAX_PAGE_HEIGHT)

                # Resize to full page for complete screenshot
                self.driver.set_window_size(self.VIEWPORT_WIDTH, page_height)
                time.sleep(0.5)

                # Save screenshot
                self.driver.save_screenshot(str(img_path))

                # Extract FULL page layout snapshot from the LIVE rendered page.
                # Saves every visible element's position, text, tag, classes,
                # computed float/display, and Range-based text line rects.
                # This allows offline bbox computation without re-crawling.
                layout = self.driver.execute_script("""
                    var content = document.getElementById('mw-content-text')
                        || document.getElementById('bodyContent')
                        || document.body;
                    var sx = window.scrollX, sy = window.scrollY;
                    var pageW = document.documentElement.scrollWidth;
                    var pageH = document.documentElement.scrollHeight;
                    
                    var elements = [];
                    var allEls = content.querySelectorAll('*');
                    
                    for (var i = 0; i < allEls.length; i++) {
                        var el = allEls[i];
                        if (el.offsetHeight === 0 && el.offsetWidth === 0) continue;
                        
                        var tag = el.tagName;
                        var text = el.textContent ? el.textContent.trim() : '';
                        if (text.length === 0 && tag !== 'IMG' && tag !== 'FIGURE') continue;
                        
                        var bcr = el.getBoundingClientRect();
                        var cs = window.getComputedStyle(el);
                        
                        var entry = {
                            tag: tag,
                            id: el.id || '',
                            cls: el.className && typeof el.className === 'string' ? el.className.substring(0, 200) : '',
                            text: text.substring(0, 500),
                            bcr: [
                                Math.round(bcr.left + sx), Math.round(bcr.top + sy),
                                Math.round(bcr.right + sx), Math.round(bcr.bottom + sy)
                            ],
                            float: cs.cssFloat || cs.float || 'none',
                            display: cs.display,
                            position: cs.position
                        };
                        
                        // For text-bearing elements, also save Range-based line rects
                        var textTags = {P:1,TD:1,TH:1,LI:1,DD:1,DT:1,CAPTION:1,FIGCAPTION:1,SPAN:1,A:1,H1:1,H2:1,H3:1,H4:1,H5:1,H6:1};
                        if (textTags[tag] && text.length > 0) {
                            try {
                                var range = document.createRange();
                                range.selectNodeContents(el);
                                var rects = range.getClientRects();
                                var lineRects = [];
                                for (var j = 0; j < rects.length; j++) {
                                    var r = rects[j];
                                    if (r.width > 3 && r.height > 2) {
                                        lineRects.push([
                                            Math.round(r.left + sx), Math.round(r.top + sy),
                                            Math.round(r.right + sx), Math.round(r.bottom + sy)
                                        ]);
                                    }
                                }
                                if (lineRects.length > 0) {
                                    entry.line_rects = lineRects;
                                }
                            } catch(e) {}
                        }
                        
                        // For images, save src
                        if (tag === 'IMG') {
                            entry.src = (el.src || '').substring(0, 200);
                            entry.text = el.alt || '';
                        }
                        
                        elements.push(entry);
                    }
                    
                    return {
                        page_width: pageW,
                        page_height: pageH,
                        num_elements: elements.length,
                        elements: elements
                    };
                """)

                # Save full layout as JSON
                layout_path = self.output_dir / f"{fname}_layout.json"
                with open(layout_path, "w") as lf:
                    json.dump(layout, lf)

                # Also compute and save bboxes using current logic (for convenience)
                para_bboxes = _compute_bboxes_from_layout(layout)
                bboxes_path = self.output_dir / f"{fname}_bboxes.json"
                with open(bboxes_path, "w") as bf:
                    json.dump(para_bboxes, bf)

                # Save HTML for reference
                html_source = self.driver.page_source
                html_path.write_text(html_source, encoding="utf-8")

                # Verify screenshot quality
                img = Image.open(img_path)
                if img.size[0] < 100 or img.size[1] < 100:
                    raise ValueError("Screenshot too small, likely failed to load")

                metadata = {
                    "entity": entity_name,
                    "url": url,
                    "screenshot_path": str(img_path),
                    "html_path": str(html_path),
                    "layout_path": str(layout_path),
                    "bboxes_path": str(bboxes_path),
                    "width": img.size[0],
                    "height": img.size[1],
                    "num_paragraphs": len(para_bboxes),
                    "num_layout_elements": layout.get("num_elements", 0),
                }
                return metadata

            except (TimeoutException, WebDriverException, ValueError) as e:
                logger.warning(f"Attempt {attempt+1}/{retries} failed for '{entity_name}': {e}")
                time.sleep(2 * (attempt + 1))

        logger.error(f"Failed to capture '{entity_name}' after {retries} retries")
        return None

    def _load_existing_metadata(self, entity_name, img_path, html_path):
        img = Image.open(img_path)
        fname = self._safe_filename(entity_name)
        bboxes_path = self.output_dir / f"{fname}_bboxes.json"
        layout_path = self.output_dir / f"{fname}_layout.json"
        meta = {
            "entity": entity_name,
            "url": self._entity_to_url(entity_name),
            "screenshot_path": str(img_path),
            "html_path": str(html_path),
            "width": img.size[0],
            "height": img.size[1],
        }
        if layout_path.exists():
            meta["layout_path"] = str(layout_path)
        if bboxes_path.exists():
            meta["bboxes_path"] = str(bboxes_path)
        return meta

    def close(self):
        self.driver.quit()


# ---------------------------------------------------------------------------
# 3. Bounding box annotation via HTML element matching
# ---------------------------------------------------------------------------

class BoundingBoxAnnotator:
    """Generate bounding boxes by locating supporting facts in rendered HTML."""

    def __init__(self, driver: webdriver.Chrome):
        self.driver = driver

    def annotate(
        self,
        entity_name: str,
        sentence_ids: List[int],
        html_path: str,
        sentences: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        For a given entity page, locate the visual bounding boxes of each
        supporting-fact sentence.

        Args:
            entity_name: Wikipedia entity title
            sentence_ids: list of sentence indices (0-based) from 2WikiMultiHopQA
            html_path: path to saved HTML of the page
            sentences: optional list of actual sentence texts from context

        Returns:
            List of {"sentence_id": int, "bbox": [x1, y1, x2, y2]} dicts
        """
        # Load the saved page in the driver
        self.driver.get(f"file://{os.path.abspath(html_path)}")
        time.sleep(0.5)

        bboxes = []
        for sid in sentence_ids:
            # Try text-based matching first if we have the sentence text
            bbox = None
            if sentences and sid < len(sentences):
                sent_text = sentences[sid].strip()
                if sent_text:
                    bbox = self._locate_by_text_match(sent_text)

            # Fallback to index-based matching
            if bbox is None:
                bbox = self._locate_sentence_bbox(sid)

            if bbox and self._validate_bbox(bbox):
                bboxes.append({"sentence_id": sid, "bbox": bbox})
            else:
                logger.warning(
                    f"Could not locate sentence {sid} for entity '{entity_name}'"
                )
        return bboxes

    def _locate_by_text_match(self, text: str) -> Optional[List[int]]:
        """Locate a sentence by matching its text content in the rendered page.
        Searches paragraph-level elements first, then walks up to the nearest
        block parent if needed."""
        try:
            escaped = text[:150].replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
            result = self.driver.execute_script("""
                var searchText = arguments[0];
                
                // Search paragraph-level elements FIRST (most precise)
                var paraEls = document.querySelectorAll("p, td, th, li, dd, dt, caption, figcaption");
                for (var i = 0; i < paraEls.length; i++) {
                    var el = paraEls[i];
                    if (el.textContent.indexOf(searchText) !== -1 && el.offsetHeight > 0) {
                        var rect = el.getBoundingClientRect();
                        if (rect.width > 50 && rect.height > 5 && rect.height < 800) {
                            return {
                                bbox: [
                                    Math.round(rect.left + window.scrollX),
                                    Math.round(rect.top + window.scrollY),
                                    Math.round(rect.right + window.scrollX),
                                    Math.round(rect.bottom + window.scrollY)
                                ]
                            };
                        }
                    }
                }
                return null;
            """, escaped)
            if result:
                return result["bbox"]
            return None
        except Exception as e:
            logger.debug(f"Text match failed: {e}")
            return None

    def _locate_sentence_bbox(self, sentence_id: int) -> Optional[List[int]]:
        """
        Locate a sentence in the rendered page by its index within the main
        content paragraphs and return its bounding box [x1, y1, x2, y2].
        """
        try:
            # Get all paragraph elements in main content
            result = self.driver.execute_script("""
                var content = document.getElementById('mw-content-text') 
                    || document.getElementById('bodyContent')
                    || document.body;
                var paragraphs = content.querySelectorAll('p, td, th, li, dd, dt, caption');
                var sentences = [];
                
                paragraphs.forEach(function(p) {
                    var text = p.textContent.trim();
                    if (text.length > 0) {
                        sentences.push({
                            text: text,
                            rect: p.getBoundingClientRect()
                        });
                    }
                });
                
                if (arguments[0] < sentences.length) {
                    var s = sentences[arguments[0]];
                    var rect = s.rect;
                    return {
                        bbox: [
                            Math.round(rect.left + window.scrollX),
                            Math.round(rect.top + window.scrollY),
                            Math.round(rect.right + window.scrollX),
                            Math.round(rect.bottom + window.scrollY)
                        ],
                        text: s.text.substring(0, 200)
                    };
                }
                return null;
            """, sentence_id)

            if result:
                return result["bbox"]
            return None

        except Exception as e:
            logger.warning(f"Error locating sentence {sentence_id}: {e}")
            return None

    def _validate_bbox(self, bbox: List[int]) -> bool:
        """Validate bounding box has positive area and reasonable dimensions."""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return False
        if w > 2000 or h > 2000:
            return False
        return True


# ---------------------------------------------------------------------------
# 4. Quality assurance
# ---------------------------------------------------------------------------

def validate_sample(sample: dict, entity_meta: Dict[str, dict]) -> bool:
    """Check that all supporting facts have valid bounding boxes."""
    evidence_chain = sample.get("evidence_chain", [])
    if not evidence_chain:
        return False
    for ev in evidence_chain:
        entity = ev.get("entity")
        if entity not in entity_meta:
            return False
        if not ev.get("bboxes"):
            return False
        # Check bboxes are within screenshot bounds
        meta = entity_meta[entity]
        w, h = meta["width"], meta["height"]
        for bbox in ev["bboxes"]:
            x1, y1, x2, y2 = bbox
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                return False
    return True


# ---------------------------------------------------------------------------
# 5. Main pipeline
# ---------------------------------------------------------------------------

def _crawl_worker(args):
    """Worker function for parallel crawling. Each worker gets its own browser."""
    entity_batch, screenshot_dir, worker_id = args
    crawler = WikiScreenshotCrawler(screenshot_dir, headless=True)
    results = {}
    for entity in entity_batch:
        meta = crawler.capture(entity)
        if meta:
            results[entity] = meta
    crawler.close()
    return results


def _annotate_from_cache(entity_name, sentence_ids, bboxes_path, sentences):
    """
    Fast annotation using pre-extracted paragraph bboxes (no browser needed).
    For each target sentence, compute similarity against all page paragraphs,
    rank them, and take the top-1 match. Discard if similarity is too low.
    """
    with open(bboxes_path, "r") as f:
        para_bboxes = json.load(f)

    import re
    from collections import Counter

    def _tokenize(text):
        """Tokenize into lowercased word tokens."""
        return re.findall(r'[a-z0-9]+', text.lower())

    def _similarity(sent_text, para_text):
        """
        Compute similarity between target sentence and a paragraph.
        Uses a combination of:
        1. Token-level F1 (handles rephrasing)
        2. Character n-gram overlap (handles partial edits)
        """
        sent_tokens = Counter(_tokenize(sent_text))
        para_tokens = Counter(_tokenize(para_text))

        # Token F1
        common = sum((sent_tokens & para_tokens).values())
        if common == 0:
            return 0.0
        precision = common / max(sum(para_tokens.values()), 1)
        recall = common / max(sum(sent_tokens.values()), 1)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # Character 4-gram overlap (catches partial matches even with reworded text)
        def char_ngrams(text, n=4):
            text = text.lower()
            return set(text[i:i+n] for i in range(len(text) - n + 1))

        sent_ng = char_ngrams(sent_text)
        para_ng = char_ngrams(para_text)
        if sent_ng:
            ng_recall = len(sent_ng & para_ng) / len(sent_ng)
        else:
            ng_recall = 0

        # Combined score: weighted average
        return 0.6 * f1 + 0.4 * ng_recall

    MIN_SIMILARITY = 0.25  # below this, discard the match

    results = {}
    for sid in sentence_ids:
        if not sentences or sid >= len(sentences):
            # No sentence text: fall back to first wide paragraph
            for p in para_bboxes:
                x1, y1, x2, y2 = p["bbox"]
                if (x2 - x1) >= 200 and len(p["text"]) > 50:
                    results[sid] = p["bbox"]
                    break
            continue

        sent_text = sentences[sid].strip()
        if not sent_text:
            continue

        # Score all paragraphs by similarity
        scored = []
        for para in para_bboxes:
            x1, y1, x2, y2 = para["bbox"]
            w, h = x2 - x1, y2 - y1
            if h < 5 or w < 50:
                continue
            sim = _similarity(sent_text, para["text"])
            scored.append((sim, w, para["bbox"], para["text"]))

        scored.sort(key=lambda x: (-x[0], -x[1]))  # best similarity first

        if scored and scored[0][0] >= MIN_SIMILARITY:
            results[sid] = scored[0][2]

    return results


def _annotate_worker(args):
    """Worker function for parallel bbox annotation. Each worker gets its own browser."""
    entity_items, screenshot_dir, worker_id = args
    # entity_items: list of (entity_name, sorted_sids, html_path, sentences)
    crawler = WikiScreenshotCrawler(screenshot_dir, headless=True)
    annotator = BoundingBoxAnnotator(crawler.driver)
    results = {}
    for entity_name, sorted_sids, html_path, sentences in entity_items:
        bboxes_info = annotator.annotate(entity_name, sorted_sids, html_path, sentences)
        sid_to_bbox = {}
        for b in bboxes_info:
            sid_to_bbox[b["sentence_id"]] = b["bbox"]
        results[entity_name] = sid_to_bbox
    crawler.close()
    return results


def build_wiki_coe(
    source_file: str,
    output_dir: str,
    screenshot_dir: str,
    max_entities: Optional[int] = None,
    num_workers: int = 4,
):
    """
    End-to-end Wiki-CoE construction pipeline.
    
    Args:
        source_file: path to 2WikiMultiHopQA JSON file
        output_dir: where to write the final dataset
        screenshot_dir: where to store screenshots
        max_entities: optional cap on number of entities to crawl
        num_workers: number of parallel browser instances for crawling
    """
    logger.info(f"Loading source dataset from {source_file}")
    with open(source_file, "r") as f:
        raw_data = json.load(f)
    logger.info(f"Loaded {len(raw_data)} questions")

    # Step 1: Rank entities by frequency
    ranked = rank_entities_by_frequency(raw_data)
    entities_to_crawl = [e for e, _ in ranked]
    if max_entities:
        entities_to_crawl = entities_to_crawl[:max_entities]
    logger.info(f"Will crawl {len(entities_to_crawl)} entities with {num_workers} workers")

    # Step 2: Crawl screenshots (parallel)
    entity_meta: Dict[str, dict] = {}

    if num_workers > 1:
        from multiprocessing import Pool
        # Split entities into batches for each worker
        batch_size = (len(entities_to_crawl) + num_workers - 1) // num_workers
        batches = []
        for i in range(num_workers):
            start = i * batch_size
            end = min(start + batch_size, len(entities_to_crawl))
            if start < end:
                batches.append((entities_to_crawl[start:end], screenshot_dir, i))

        logger.info(f"Starting {len(batches)} parallel workers...")
        with Pool(processes=len(batches)) as pool:
            results = pool.map(_crawl_worker, batches)
        for r in results:
            entity_meta.update(r)
    else:
        crawler = WikiScreenshotCrawler(screenshot_dir, headless=True)
        for entity in tqdm(entities_to_crawl, desc="Crawling screenshots"):
            meta = crawler.capture(entity)
            if meta:
                entity_meta[entity] = meta
        crawler.close()

    logger.info(f"Successfully crawled {len(entity_meta)}/{len(entities_to_crawl)} entities")

    # Build context lookup: entity -> list of sentences
    # context format: [[title, [sent0, sent1, ...]], ...]
    logger.info("Building context sentence lookup...")
    global_context: Dict[str, list] = {}
    for sample in raw_data:
        for ctx_entry in sample.get("context", []):
            if isinstance(ctx_entry, list) and len(ctx_entry) == 2:
                title, sents = ctx_entry
                if title not in global_context:
                    global_context[title] = sents

    # Step 3: Pre-annotate all unique (entity, sentence_id) pairs ONCE
    logger.info("Collecting unique (entity, sentence_id) pairs...")
    entity_sids: Dict[str, set] = {}
    relevant_questions = []
    for sample in raw_data:
        supporting_facts = sample.get("supporting_facts", [])
        facts_for_q: Dict[str, List[int]] = {}
        if isinstance(supporting_facts, list):
            for pair in supporting_facts:
                if isinstance(pair, list) and len(pair) == 2:
                    title, sid = pair[0], pair[1]
                    if title in entity_meta:
                        facts_for_q.setdefault(title, []).append(sid)
                        entity_sids.setdefault(title, set()).add(sid)

        if facts_for_q:
            relevant_questions.append((sample, facts_for_q))

    total_pairs = sum(len(sids) for sids in entity_sids.values())
    logger.info(
        f"Found {len(relevant_questions)} relevant questions, "
        f"{len(entity_sids)} entities, {total_pairs} unique (entity, sid) pairs to annotate"
    )

    # Annotate each entity's sentences
    # Use pre-extracted bboxes if available (fast, no browser), otherwise use Selenium
    anno_items = []
    fast_items = []
    for entity, sids in entity_sids.items():
        meta = entity_meta[entity]
        sentences = global_context.get(entity, None)
        sorted_sids = sorted(sids)
        if "bboxes_path" in meta and os.path.exists(meta["bboxes_path"]):
            fast_items.append((entity, sorted_sids, meta["bboxes_path"], sentences))
        else:
            anno_items.append((entity, sorted_sids, meta["html_path"], sentences))

    bbox_cache: Dict[str, Dict[int, Optional[List[int]]]] = {}

    # Fast annotation from pre-extracted bboxes (no browser)
    if fast_items:
        logger.info(f"Fast-annotating {len(fast_items)} entities from cached bboxes...")
        for entity, sorted_sids, bboxes_path, sentences in tqdm(fast_items, desc="Fast annotating"):
            bbox_cache[entity] = _annotate_from_cache(entity, sorted_sids, bboxes_path, sentences)

    # Slow annotation via Selenium for entities without cached bboxes
    if anno_items:
        if num_workers > 1:
            from multiprocessing import Pool
            anno_batch_size = (len(anno_items) + num_workers - 1) // num_workers
            anno_batches = []
            for i in range(num_workers):
                start = i * anno_batch_size
                end = min(start + anno_batch_size, len(anno_items))
                if start < end:
                    anno_batches.append((anno_items[start:end], screenshot_dir, 300 + i))

            logger.info(f"Selenium-annotating {len(anno_items)} entities with {len(anno_batches)} parallel workers...")
            with Pool(processes=len(anno_batches)) as pool:
                anno_results = pool.map(_annotate_worker, anno_batches)
            for r in anno_results:
                bbox_cache.update(r)
        else:
            anno_crawler = WikiScreenshotCrawler(screenshot_dir, headless=True)
            annotator = BoundingBoxAnnotator(anno_crawler.driver)
            for entity, sorted_sids, html_path, sentences in tqdm(anno_items, desc="Annotating entities"):
                bboxes_info = annotator.annotate(entity, sorted_sids, html_path, sentences)
                sid_to_bbox = {}
                for b in bboxes_info:
                    sid_to_bbox[b["sentence_id"]] = b["bbox"]
                bbox_cache[entity] = sid_to_bbox
            anno_crawler.close()

    logger.info(f"Annotated {len(bbox_cache)} entities ({len(fast_items)} fast, {len(anno_items)} via browser)")

    # Step 4: Assemble dataset from cache (fast, no browser needed)
    dataset = []
    for sample, facts_for_q in tqdm(relevant_questions, desc="Assembling dataset"):
        evidence_chain = []
        all_valid = True

        for entity, sids in facts_for_q.items():
            meta = entity_meta[entity]
            cached = bbox_cache.get(entity, {})
            bboxes = [cached[sid] for sid in sids if sid in cached]

            if not bboxes:
                all_valid = False
                break

            evidence_chain.append({
                "entity": entity,
                "screenshot": os.path.basename(meta["screenshot_path"]),
                "width": meta["width"],
                "height": meta["height"],
                "bboxes": bboxes,
                "sentence_ids": sids,
            })

        if not all_valid or not evidence_chain:
            continue

        coe_sample = {
            "question": sample["question"],
            "answer": sample["answer"],
            "type": sample.get("type", "unknown"),
            "num_hops": len(evidence_chain),
            "evidence_chain": evidence_chain,
        }

        if validate_sample(coe_sample, entity_meta):
            dataset.append(coe_sample)

    # Step 4: Split into train/test
    logger.info(f"Total valid samples: {len(dataset)}")
    train_size = min(40000, int(len(dataset) * 0.8))
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]

    # Save
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_data, f, indent=2)
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(test_data, f, indent=2)

    # Save entity metadata
    with open(os.path.join(output_dir, "entity_meta.json"), "w") as f:
        json.dump(entity_meta, f, indent=2)

    logger.info(f"Saved {len(train_data)} train, {len(test_data)} test samples to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build Wiki-CoE dataset")
    parser.add_argument("--source", required=True, help="Path to 2WikiMultiHopQA JSON")
    parser.add_argument("--output_dir", default="data/wiki_coe", help="Output directory")
    parser.add_argument("--screenshot_dir", default="data/wiki_coe/screenshots")
    parser.add_argument("--max_entities", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4, help="Parallel browser instances")
    args = parser.parse_args()

    build_wiki_coe(args.source, args.output_dir, args.screenshot_dir, args.max_entities, args.num_workers)
