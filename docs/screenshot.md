# Screenshot Feature

The JAXS physics simulator includes a built-in screenshot feature for capturing the current scene.

## Usage

Press `P` to take a screenshot while the simulation is running.

## Details

- Screenshots are saved to the `screenshots/` folder in the project root
- Files are named with timestamp: `screenshot_YYYYMMDD_HHMMSS_milliseconds.png`
- The system logs when a screenshot is taken and saved
- Images are saved in PNG format with full color

## Key Bindings

| Key | Action |
|-----|---------|
| P | Take screenshot |
| F | Toggle fullscreen |
| Escape | Release mouse capture |
| W/A/S/D | Move camera |
| Space | Move up |
| Left Shift | Move down |
| Mouse | Look around |

## Technical Notes

- Screenshots capture the current framebuffer contents
- Images are automatically flipped to correct GPU orientation
- Processing happens asynchronously to avoid blocking the render loop
- A 2-second indicator is shown when a screenshot is taken (in logs)