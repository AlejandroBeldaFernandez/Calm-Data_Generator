# Stream Generator Module
from calm_data_generator.generators.stream.StreamGenerator import StreamGenerator
from calm_data_generator.generators.stream.StreamReporter import StreamReporter

# StreamBlockGenerator may have a different class name after renaming
try:
    from calm_data_generator.generators.stream.StreamBlockGenerator import (
        StreamBlockGenerator,
    )
except ImportError:
    # Try alternative import
    try:
        from calm_data_generator.generators.stream.StreamBlockGenerator import (
            SyntheticBlockGenerator as StreamBlockGenerator,
        )
    except ImportError:
        StreamBlockGenerator = None

__all__ = ["StreamGenerator", "StreamReporter", "StreamBlockGenerator"]
