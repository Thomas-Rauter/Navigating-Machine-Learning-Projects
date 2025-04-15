from PIL import Image

# Load your texture tile
tile = Image.open("../../figures/general/ml_cover_texture_tile.png")

# A4 dimensions in pixels at 300 DPI
a4_width_px, a4_height_px = int(8.27 * 300), int(11.69 * 300)

# Create a blank A4 image with RGBA
background = Image.new("RGBA", (a4_width_px, a4_height_px), (0, 0, 0, 0))

# Paste tiles to cover the full A4 area
for x in range(0, a4_width_px, tile.width):
    for y in range(0, a4_height_px, tile.height):
        background.paste(tile, (x, y))

# Save final background
background.convert("RGB").save("a4_texture_background.png", dpi=(300, 300))
