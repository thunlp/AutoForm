from agentverse.registry import Registry

decision_maker_registry = Registry(name="DecisionMakerRegistry")

from .base import BaseDecisionMaker, DummyDecisionMaker
from .horizontal import HorizontalDecisionMaker
from .vertical import VerticalDecisionMaker
from .dynamic import DynamicDecisionMaker
from .vertical_solver_first import VerticalSolverFirstDecisionMaker
from .concurrent import ConcurrentDecisionMaker
from .horizontal_tool import HorizontalToolDecisionMaker
from .central import CentralDecisionMaker
from .brainstorming import BrainstormingDecisionMaker
from .debate import DebateDecisionMaker
from .debate_internal import DebateInternalDecisionMaker
from .debate_hotpotqa import DebateHotpotqaDecisionMaker
from .debate_hotpotqa_internal import DebateHotpotqaInternalDecisionMaker
from .debate_word_guessing import DebateWordGuessingDecisionMaker
