# threshold for significance testing
P_SIG: float = 0.05
# warn the user if they pick a value larger than this
P_SIG_WARNING: float = 0.1
SAMPLE_CNT_MIN = 100
EXPERIMENT_CNT_MIN = 3

# emojis
EMOJIS_SUCCESS: dict = {True: "✅", False: "❌"}
EMOJIS_THUMBS = {"up": "👍", "down": "👎"}

# object representation
OBJECT_REPR_PRECISION: int = 2
OBJECT_REPR_METRIC_CNT: int = 3

# viz
P_LOW: int = 25  # lower bound: 25th percentile for drawing metric representation
P_HIGH: int = 75  # upper bound: 75th percentile for drawing metric representation
