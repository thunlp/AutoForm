from agentverse.registry import Registry

dataloader_registry = Registry(name="dataloader")

from .gsm8k import GSM8KLoader
from .responsegen import ResponseGenLoader
from .humaneval import HumanevalLoader
from .commongen import CommongenLoader
from .mgsm import MGSMLoader
from .logic_grid import LogicGridLoader, LogicGridManyTaskDescriptionLoader
from .matrix_shape import MatrixShapeLoader, MatrixShapeManyTaskDescriptionLoader
from .hotpot_qa import HotpotQALoader
from .aqua import AquaDataloader, AquaManyTaskDescriptionDataloader
from .standard import StandardDataloader
from .qa import QALoader