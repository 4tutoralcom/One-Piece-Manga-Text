from bubble_detector import BubbleDetector

detector = BubbleDetector()

detector.debug(
    "input/012.png",
    "output/debug_page012.png",
)

detector.crop(
    "input/012.png",
    "output/debug_page012/",
)
detections=detector.crop_masks(
    "input/012.png",
    "output/debug_page012_crop_masks/",
)

detector.save_json(detections, "output/debug_page012.json")
