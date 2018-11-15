# music21 is an optional dep
from ...core import get_logger
logger = get_logger()

try:
    from .music import pitch_and_duration_to_quantized
    from .music import pitches_and_durations_to_pretty_midi
    from .music import quantized_to_pretty_midi
    from .music import quantized_to_pitch_duration
    from .music import plot_pitches_and_durations
    from .music import music21_to_pitch_duration
    from .music import music21_to_quantized
    from .music import plot_piano_roll
    from .music import quantized_imlike_to_image_array
    from .analysis import midi_to_notes
    from .analysis import notes_to_midi
    from .loaders import fetch_jsb
    from .loaders import fetch_josquin
except ImportError:
    logger.info("Unable to import music21 related utilities")
