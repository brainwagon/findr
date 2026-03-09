import tetra3
from PIL import Image
import sys
import os

def verify():
    print("Initializing Tetra3...")
    try:
        t3 = tetra3.Tetra3()
    except Exception as e:
        print(f"Failed to initialize Tetra3: {e}")
        return

    sample_image_path = 'test-images/lores_jpeg_2025-11-07T03_03_46.674Z.jpg'
    if not os.path.exists(sample_image_path):
        print(f"Sample image not found: {sample_image_path}")
        return

    print(f"Loading image: {sample_image_path}")
    img = Image.open(sample_image_path)

    print("Attempting to solve...")
    # These lores images are quite small, FOV estimate might help.
    # The README mentioned a prototype telescope finder.
    # I'll try without first.
    solution = t3.solve_from_image(img)

    if solution['RA'] is not None:
        print("Solved!")
        print(f"RA: {solution['RA']}")
        print(f"Dec: {solution['Dec']}")
        print(f"Roll: {solution['Roll']}")
        print(f"FOV: {solution['FOV']}")
    else:
        print("Failed to solve image.")
        print(f"Centroids found: {len(solution.get('centroids', []))}")

if __name__ == "__main__":
    verify()
