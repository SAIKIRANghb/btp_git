# Ball Tracking and Trajectory Prediction

## Overview

This project implements a ball tracking and trajectory prediction system using OpenCV. It detects a red ball in a video, tracks its movement, predicts future positions, and detects successful hits on a defined bat region.

## Features

- **Ball Detection**: Uses HSV color filtering to detect a red ball in the video.
- **Trajectory Tracking**: Stores the ball's past positions for analysis.
- **Hit Detection**: Identifies when the ball crosses a bat region.
- **Trajectory Prediction**: Estimates future positions based on velocity and gravity.
- **Playback Speed Control**: Allows adjusting video playback speed.
- **User Interaction**: Toggle tracking, switch modes, reset hits, and adjust playback speed.

## Installation

1. Install dependencies:
   ```sh
   pip install opencv-python numpy
   ```
2. Place your video file (`red_ball1.mp4`) in the project directory.
3. Run the script:
   ```sh
   python ball_tracker.py
   ```

## Controls

- `q` - Quit program
- `m` - Toggle tracking/prediction mode
- `t` - Start/Stop ball tracking
- `r` - Reset hit count and trajectory
- `+` - Increase playback speed
- `-` - Decrease playback speed

## Dependencies

- Python 3
- OpenCV
- NumPy


