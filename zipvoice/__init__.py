import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="pkg_resources is deprecated as an API.*",
)
