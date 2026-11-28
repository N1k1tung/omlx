from dataclasses import dataclass

from ....models.ernie5_5_moe.config import ModelConfig as TextConfig
from ..deepseek_v4_dspark.config import DeepseekV4DsparkConfig


@dataclass
class ModelConfig(DeepseekV4DsparkConfig):
    """ERNIE 5.5 Flash DSpark head; the proposal fields mirror ``DSparkConfig``."""

    model_type: str = "ernie5_5_dspark"

    def __post_init__(self):
        if isinstance(self.text_config, dict):
            self.text_config = TextConfig.from_dict(self.text_config)
        super().__post_init__()
