# SportVU Shot Detector

A Python-based shot detection and quality scoring system built on NBA SportVU optical tracking data.

The program processes raw SportVU JSON game files, scanning every tracked frame to identify shot attempts, classify them by type (dunk, layup, 2pt jump shot, 3pt jump shot), assign each shot to a court zone, and compute a shot quality score based on zone efficiency and defensive contest level. Results are exported to a CSV log and visualized on a shot quality timeline.

For a full explanation of the detection methodology, quality metric design, and results, see [abstract.pdf](abstract.pdf).

## Usage

The program was run against a single SportVU JSON file for the January 2, 2016 Brooklyn Nets @ Boston Celtics game (game ID `0021500495`). Due to the size of the raw SportVU JSON files, the data is not included in this repository. SportVU tracking data of this format is available through various NBA data repositories online.
