from pathlib import Path
from tempfile import TemporaryDirectory

from docling.document_converter import DocumentConverter

from bubble_detector import BubbleDetector

MANGA_ROOT = Path("/app/manga")
TEXT_ROOT = Path("/app/text")

MIN_TEXT_LENGTH = 20

converter = DocumentConverter()
detector = BubbleDetector()


def run_docling(image_path: Path) -> str:
    try:
        result = converter.convert(str(image_path))
        return result.document.export_to_text().strip()
    except Exception as e:
        print(f"Docling failed on {image_path.name}: {e}")
        return ""


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
}


for chapter_dir in sorted(MANGA_ROOT.iterdir()):

    if not chapter_dir.is_dir():
        continue

    print(f"\n===== {chapter_dir.name} =====", flush=True)

    output_chapter = TEXT_ROOT / chapter_dir.name
    output_chapter.mkdir(parents=True, exist_ok=True)

    images = sorted(
        p
        for p in chapter_dir.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image in images:

        output_file = output_chapter / f"{image.stem}.txt"

        # Skip pages that already have text
        if output_file.exists():

            existing = output_file.read_text(
                encoding="utf-8",
                errors="ignore",
            ).strip()

            if len(existing) >= MIN_TEXT_LENGTH:
                print(f"Skipping {image.name}")
                continue

        print(f"Processing {image.name}", flush=True)

        with TemporaryDirectory() as tmpdir:
            print(f"  Detecting bubbles in {image.name} using {tmpdir}", flush=True)

            tmpdir = Path(tmpdir)

            # Detect speech bubbles
            detections = detector.crop_masks(
                image,
                tmpdir,
            )

            if not detections:
                print("  No bubbles detected.")

                output_file.write_text(
                    "",
                    encoding="utf-8",
                )

                continue

            page_text = []

            # OCR every detected bubble
            for bubble in sorted(
                detections,
                key=lambda b: (b["y"], b["x"]),
            ):

                bubble_file = tmpdir / bubble["file"]

                print(
                    f"    OCR {bubble_file.name}",
                    flush=True,
                )

                text = run_docling(bubble_file)

                if text:
                    page_text.append(text)

            final_text = "\n\n".join(page_text).strip()

            output_file.write_text(
                final_text,
                encoding="utf-8",
            )

            print(
                f"  Saved {output_file.name} ({len(final_text)} chars)",
                flush=True,
            )

print("\nFinished.")