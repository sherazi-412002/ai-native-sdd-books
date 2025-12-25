---
sidebar_position: 2
---

# LLM Cognitive Planning

## Cognitive Planning with Large Language Models

This chapter covers using Large Language Models (LLMs) for cognitive planning in humanoid robots. Cognitive planning refers to the ability of an AI system to understand high-level goals, decompose them into executable tasks, reason about the world state, and generate appropriate action sequences to achieve objectives. For humanoid robots, this involves processing natural language commands and translating them into complex behavioral sequences.

LLM-based cognitive planning leverages the reasoning capabilities, world knowledge, and language understanding of large language models to create more flexible and adaptive robotic systems. Unlike traditional planning approaches that rely on pre-defined rules and symbolic representations, LLMs can handle ambiguous, natural language instructions and generate plans for novel situations.

## Introduction to Cognitive Planning

Cognitive planning in robotics involves several key components:

1. **Goal Understanding**: Interpreting high-level goals or commands
2. **World Modeling**: Maintaining an internal representation of the environment
3. **Plan Generation**: Creating sequences of actions to achieve goals
4. **Plan Execution**: Executing actions while monitoring progress
5. **Adaptation**: Adjusting plans based on feedback and changing conditions

```python
import openai
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import asyncio
import time
from enum import Enum

@dataclass
class WorldState:
    """Represents the current state of the world"""
    objects: Dict[str, Dict]  # Object name to properties
    robot_state: Dict[str, Any]  # Robot's current state
    environment: Dict[str, Any]  # Environmental conditions
    time: float  # Current simulation time
    location: str  # Current location of robot

@dataclass
class Action:
    """Represents an action that can be executed"""
    name: str
    parameters: Dict[str, Any]
    duration: float  # Expected duration in seconds
    preconditions: List[str]  # Conditions that must be true before execution
    effects: List[str]  # Effects of the action on the world state

@dataclass
class PlanStep:
    """A step in a plan"""
    action: Action
    expected_outcome: str
    confidence: float
    dependencies: List[int]  # Indices of steps that must be completed first

@dataclass
class CognitivePlan:
    """A complete cognitive plan"""
    steps: List[PlanStep]
    goal: str
    context: Dict[str, Any]
    creation_time: float
    estimated_duration: float

class PlanningState(Enum):
    """State of the planning process"""
    IDLE = 0
    UNDERSTANDING_GOAL = 1
    ANALYZING_WORLD = 2
    GENERATING_PLAN = 3
    VALIDATING_PLAN = 4
    EXECUTING = 5
    ADAPTING = 6
    COMPLETED = 7
    FAILED = 8

class CognitivePlanner:
    """Main cognitive planning system using LLMs"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.api_key = api_key
        self.model = model
        self.current_state = WorldState(
            objects={},
            robot_state={},
            environment={},
            time=time.time(),
            location="unknown"
        )
        self.current_plan = None
        self.planning_state = PlanningState.IDLE
        self.action_repository = self._initialize_action_repository()

    def _initialize_action_repository(self) -> Dict[str, Dict]:
        """Initialize available actions for the robot"""
        return {
            "move_to": {
                "description": "Move the robot to a specific location",
                "parameters": {
                    "target_location": "str - The destination location",
                    "speed": "float - Movement speed (0.1-1.0)"
                },
                "preconditions": ["robot_is_active", "target_location_is_known"],
                "effects": ["robot_location_changes"]
            },
            "pick_up": {
                "description": "Pick up an object with the robot's manipulator",
                "parameters": {
                    "object_name": "str - Name of the object to pick up",
                    "arm": "str - Which arm to use (left, right)"
                },
                "preconditions": ["object_is_reachable", "object_is_graspable", "arm_is_free"],
                "effects": ["object_is_held", "arm_is_occupied"]
            },
            "place": {
                "description": "Place an object at a specific location",
                "parameters": {
                    "object_name": "str - Name of the object to place",
                    "target_location": "str - Where to place the object",
                    "arm": "str - Which arm is holding the object"
                },
                "preconditions": ["object_is_held", "target_location_is_reachable"],
                "effects": ["object_is_placed", "arm_is_free"]
            },
            "navigate_to_object": {
                "description": "Navigate to an object's location",
                "parameters": {
                    "object_name": "str - Name of the target object"
                },
                "preconditions": ["object_is_visible", "object_location_is_known"],
                "effects": ["robot_is_near_object"]
            },
            "detect_object": {
                "description": "Detect and identify objects in the environment",
                "parameters": {
                    "object_type": "str - Type of object to detect (optional)"
                },
                "preconditions": ["camera_is_active"],
                "effects": ["object_locations_updated"]
            },
            "ask_human": {
                "description": "Ask for clarification or information from a human",
                "parameters": {
                    "question": "str - The question to ask",
                    "target_person": "str - Who to ask (optional)"
                },
                "preconditions": ["human_is_present"],
                "effects": ["information_acquired"]
            }
        }

    async def understand_goal(self, goal_description: str) -> Dict[str, Any]:
        """Use LLM to understand and decompose a high-level goal"""
        self.planning_state = PlanningState.UNDERSTANDING_GOAL

        prompt = f"""
        Analyze the following goal and break it down into components:
        Goal: "{goal_description}"

        Provide a structured analysis with:
        1. Main objective
        2. Sub-goals or required steps
        3. Necessary conditions or prerequisites
        4. Potential obstacles or challenges
        5. Success criteria

        Respond in JSON format with keys: main_objective, sub_goals, prerequisites, challenges, success_criteria
        """

        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        analysis = json.loads(response.choices[0].message.content)
        return analysis

    async def analyze_world_state(self, goal_analysis: Dict[str, Any]) -> WorldState:
        """Analyze the current world state in relation to the goal"""
        self.planning_state = PlanningState.ANALYZING_WORLD

        # In a real system, this would interface with perception systems
        # For this example, we'll simulate world state analysis
        current_state = self.current_state

        # Update state based on goal requirements
        required_objects = self._extract_required_objects(goal_analysis)
        detected_objects = await self._detect_relevant_objects(required_objects)

        current_state.objects.update(detected_objects)
        current_state.time = time.time()

        return current_state

    def _extract_required_objects(self, goal_analysis: Dict[str, Any]) -> List[str]:
        """Extract object names that might be needed for the goal"""
        # This would use NLP to extract object references from the goal analysis
        text_context = f"{goal_analysis.get('main_objective', '')} {goal_analysis.get('sub_goals', '')}"
        # Simple keyword extraction (in practice, use proper NLP)
        import re
        words = re.findall(r'\b\w+\b', text_context.lower())
        potential_objects = [word for word in words if len(word) > 2]  # Filter short words
        return list(set(potential_objects))  # Remove duplicates

    async def _detect_relevant_objects(self, object_names: List[str]) -> Dict[str, Dict]:
        """Simulate object detection (in real system, this would use perception)"""
        # Simulate detection results
        detected_objects = {}
        for obj_name in object_names[:5]:  # Limit for simulation
            detected_objects[obj_name] = {
                "location": f"{obj_name}_location",
                "pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # x, y, z, roll, pitch, yaw
                "properties": {
                    "graspable": True,
                    "movable": True,
                    "size": "medium"
                },
                "confidence": 0.8 + np.random.random() * 0.2  # 0.8-1.0
            }
        return detected_objects

    async def generate_plan(self, goal_analysis: Dict[str, Any], world_state: WorldState) -> CognitivePlan:
        """Generate a cognitive plan using LLM"""
        self.planning_state = PlanningState.GENERATING_PLAN

        # Create prompt for plan generation
        prompt = f"""
        Generate a detailed plan to achieve the following goal:
        Main Objective: {goal_analysis['main_objective']}
        Sub-goals: {goal_analysis['sub_goals']}
        Prerequisites: {goal_analysis['prerequisites']}

        Current World State:
        - Robot Location: {world_state.location}
        - Available Objects: {list(world_state.objects.keys())}
        - Robot Capabilities: {list(self.action_repository.keys())}

        Create a step-by-step plan with:
        1. Specific actions from the available action set
        2. Required parameters for each action
        3. Estimated confidence for each step
        4. Dependencies between steps

        Available actions: {list(self.action_repository.keys())}
        Each action has: {self.action_repository}

        Respond in JSON format with structure:
        {{
            "steps": [
                {{
                    "action": "action_name",
                    "parameters": {{"param1": "value1", ...}},
                    "expected_outcome": "description",
                    "confidence": 0.0-1.0,
                    "dependencies": [step_indices]
                }}
            ],
            "estimated_duration": total_seconds
        }}
        """

        response = await openai.ChatCompletion.acreate(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )

        plan_data = json.loads(response.choices[0].message.content)

        # Convert to CognitivePlan object
        steps = []
        for step_data in plan_data['steps']:
            action = Action(
                name=step_data['action'],
                parameters=step_data['parameters'],
                duration=1.0,  # Default duration
                preconditions=self.action_repository[step_data['action']].get('preconditions', []),
                effects=self.action_repository[step_data['action']].get('effects', [])
            )
            step = PlanStep(
                action=action,
                expected_outcome=step_data['expected_outcome'],
                confidence=step_data['confidence'],
                dependencies=step_data['dependencies']
            )
            steps.append(step)

        plan = CognitivePlan(
            steps=steps,
            goal=goal_analysis['main_objective'],
            context=goal_analysis,
            creation_time=time.time(),
            estimated_duration=plan_data['estimated_duration']
        )

        self.current_plan = plan
        return plan

    async def validate_plan(self, plan: CognitivePlan, world_state: WorldState) -> Tuple[bool, List[str]]:
        """Validate the generated plan for feasibility"""
        self.planning_state = PlanningState.VALIDATING_PLAN

        issues = []

        # Check preconditions for each step
        for i, step in enumerate(plan.steps):
            for precondition in step.action.preconditions:
                # In a real system, this would check actual world state
                # For simulation, we'll assume some basic validation
                if "robot_is_active" in precondition and not world_state.robot_state.get("active", False):
                    issues.append(f"Step {i}: Robot is not active, required for {step.action.name}")

        # Check for conflicts between steps
        for i in range(len(plan.steps)):
            for j in range(i + 1, len(plan.steps)):
                if (i in plan.steps[j].dependencies and
                    j in plan.steps[i].dependencies):
                    issues.append(f"Circular dependency between steps {i} and {j}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def execute_plan(self, plan: CognitivePlan):
        """Execute the cognitive plan"""
        self.planning_state = PlanningState.EXECUTING

        # This would interface with the robot's execution system
        # For this example, we'll simulate execution
        for i, step in enumerate(plan.steps):
            print(f"Executing step {i}: {step.action.name} with params {step.action.parameters}")
            # Simulate execution time
            time.sleep(0.1)

            # Check if step succeeded
            success = np.random.random() > 0.1  # 90% success rate for simulation
            if not success:
                print(f"Step {i} failed, adapting plan...")
                self.planning_state = PlanningState.ADAPTING
                # In a real system, this would trigger replanning
                break

        if self.planning_state == PlanningState.EXECUTING:
            self.planning_state = PlanningState.COMPLETED

    async def process_goal(self, goal_description: str) -> Optional[CognitivePlan]:
        """Complete process from goal to execution-ready plan"""
        try:
            # Step 1: Understand the goal
            goal_analysis = await self.understand_goal(goal_description)

            # Step 2: Analyze world state
            world_state = await self.analyze_world_state(goal_analysis)

            # Step 3: Generate plan
            plan = await self.generate_plan(goal_analysis, world_state)

            # Step 4: Validate plan
            is_valid, issues = await self.validate_plan(plan, world_state)

            if not is_valid:
                print(f"Plan validation failed with issues: {issues}")
                # In a real system, this might trigger plan refinement
                return None

            return plan

        except Exception as e:
            print(f"Error in goal processing: {e}")
            self.planning_state = PlanningState.FAILED
            return None

# Example usage
async def example_cognitive_planning():
    """Example of using the cognitive planning system"""
    # This would require an actual OpenAI API key
    # For demonstration purposes, we'll show the structure
    planner = CognitivePlanner(api_key="your-api-key-here")

    goal = "Please bring me a cup of coffee from the kitchen"
    plan = await planner.process_goal(goal)

    if plan:
        print(f"Generated plan for: {plan.goal}")
        print(f"Plan has {len(plan.steps)} steps")
        for i, step in enumerate(plan.steps):
            print(f"  {i+1}. {step.action.name} - {step.expected_outcome} (confidence: {step.confidence:.2f})")

    return plan
```

## LLM Integration

Integrating LLMs into cognitive planning systems requires careful consideration of prompt engineering, response parsing, and system integration. The following implementation demonstrates how to effectively use LLMs for various planning tasks.

```python
import openai
import json
import asyncio
from typing import Dict, List, Any, Optional, Callable
import re
from pydantic import BaseModel, Field
import instructor
from enum import Enum

class TaskType(Enum):
    """Types of planning tasks"""
    GOAL_DECOMPOSITION = "goal_decomposition"
    ACTION_PLANNING = "action_planning"
    CONTEXT_ANALYSIS = "context_analysis"
    TASK_REFINEMENT = "task_refinement"
    EXECUTION_MONITORING = "execution_monitoring"

class LLMInterface:
    """Interface for LLM integration in cognitive planning"""

    def __init__(self, api_key: str, model: str = "gpt-4"):
        openai.api_key = api_key
        self.model = model
        self.client = instructor.patch(openai.ChatCompletion)

    async def generate_with_schema(self, prompt: str, response_model: BaseModel) -> BaseModel:
        """Generate response with structured schema"""
        try:
            response = await self.client.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_model=response_model,
                temperature=0.3
            )
            return response
        except Exception as e:
            print(f"Error generating with schema: {e}")
            return None

    async def generate_text(self, prompt: str, temperature: float = 0.3) -> str:
        """Generate text response from LLM"""
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating text: {e}")
            return ""

class GoalDecomposition(BaseModel):
    """Schema for goal decomposition response"""
    main_objective: str = Field(description="The main goal to be achieved")
    sub_goals: List[str] = Field(description="List of sub-goals that need to be achieved")
    priority_order: List[int] = Field(description="Order of sub-goal priority (indices of sub_goals)")
    dependencies: Dict[str, List[str]] = Field(description="Dependencies between sub-goals")
    success_criteria: List[str] = Field(description="Criteria for determining success")

class ActionPlan(BaseModel):
    """Schema for action planning response"""
    steps: List[Dict] = Field(description="List of steps to execute")
    action_name: str = Field(description="Name of the action")
    parameters: Dict[str, Any] = Field(description="Parameters for the action")
    preconditions: List[str] = Field(description="Conditions that must be true before action")
    expected_effects: List[str] = Field(description="Expected effects of the action")
    confidence: float = Field(description="Confidence in the plan (0.0-1.0)")

class ContextAnalysis(BaseModel):
    """Schema for context analysis response"""
    current_situation: str = Field(description="Current situation analysis")
    relevant_objects: List[str] = Field(description="Objects relevant to the task")
    environmental_factors: List[str] = Field(description="Environmental factors to consider")
    constraints: List[str] = Field(description="Constraints on the solution")
    opportunities: List[str] = Field(description="Opportunities for optimization")

class TaskRefinement(BaseModel):
    """Schema for task refinement response"""
    refined_task: str = Field(description="More specific version of the task")
    assumptions: List[str] = Field(description="Assumptions made during refinement")
    clarifications_needed: List[str] = Field(description="Information needed for better execution")
    alternative_approaches: List[str] = Field(description="Alternative approaches to consider")

class ExecutionMonitoring(BaseModel):
    """Schema for execution monitoring response"""
    progress: float = Field(description="Progress toward goal (0.0-1.0)")
    obstacles_encountered: List[str] = Field(description="Obstacles encountered")
    plan_adaptations: List[str] = Field(description="Suggested adaptations to the plan")
    confidence_level: float = Field(description="Confidence in continued success (0.0-1.0)")

class LLMCognitivePlanner:
    """Advanced cognitive planner using LLM integration"""

    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.task_history = []
        self.context_memory = {}

    async def decompose_goal(self, goal: str, context: Dict[str, Any] = None) -> Optional[GoalDecomposition]:
        """Decompose a high-level goal using LLM"""
        prompt = f"""
        Decompose the following goal into manageable sub-goals:

        Goal: {goal}

        Context: {context or 'No additional context provided'}

        Consider:
        1. What are the main components of this goal?
        2. What needs to happen first, second, etc.?
        3. Are there dependencies between different parts?
        4. How will we know when each part is complete?

        Provide your analysis in the required JSON format.
        """

        return await self.llm.generate_with_schema(prompt, GoalDecomposition)

    async def generate_action_plan(self, sub_goal: str, world_state: Dict[str, Any]) -> Optional[ActionPlan]:
        """Generate an action plan for a sub-goal using LLM"""
        prompt = f"""
        Generate a detailed action plan for the following sub-goal:

        Sub-goal: {sub_goal}

        Current World State:
        {json.dumps(world_state, indent=2)}

        Available Actions: navigate_to_object, pick_up, place, detect_object, ask_human, move_to

        Create a step-by-step plan that:
        1. Uses available actions
        2. Specifies necessary parameters
        3. Considers preconditions
        4. Estimates expected outcomes

        Provide your plan in the required JSON format.
        """

        return await self.llm.generate_with_schema(prompt, ActionPlan)

    async def analyze_context(self, situation: str, goal: str) -> Optional[ContextAnalysis]:
        """Analyze the current context for planning"""
        prompt = f"""
        Analyze the following situation in the context of achieving the specified goal:

        Current Situation: {situation}
        Goal: {goal}

        Provide analysis considering:
        1. What is the current state of affairs?
        2. What objects or entities are relevant?
        3. What environmental factors should be considered?
        4. What constraints limit the possible approaches?
        5. Are there opportunities for optimization or improvement?

        Format your response in the required JSON structure.
        """

        return await self.llm.generate_with_schema(prompt, ContextAnalysis)

    async def refine_task(self, task_description: str, execution_history: List[Dict] = None) -> Optional[TaskRefinement]:
        """Refine a task based on execution history or current understanding"""
        history_context = execution_history or []
        prompt = f"""
        Refine the following task description based on execution history:

        Task: {task_description}
        Execution History: {history_context}

        Provide a more specific and actionable version of the task that:
        1. Clarifies ambiguous elements
        2. Identifies assumptions being made
        3. Highlights information needed for better execution
        4. Suggests alternative approaches if the original seems problematic

        Return in the specified JSON format.
        """

        return await self.llm.generate_with_schema(prompt, TaskRefinement)

    async def monitor_execution(self, current_state: Dict[str, Any],
                              original_plan: List[Dict],
                              executed_steps: List[Dict]) -> Optional[ExecutionMonitoring]:
        """Monitor execution progress and suggest adaptations"""
        prompt = f"""
        Monitor the execution progress for the following plan:

        Original Plan: {original_plan}
        Executed Steps: {executed_steps}
        Current State: {current_state}

        Analyze:
        1. What progress has been made?
        2. What obstacles have been encountered?
        3. What adaptations to the plan are needed?
        4. How confident are we in achieving the goal?

        Provide monitoring results in the required JSON format.
        """

        return await self.llm.generate_with_schema(prompt, ExecutionMonitoring)

    async def create_comprehensive_plan(self, goal: str, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive plan using multiple LLM calls"""
        # Step 1: Analyze context
        context_analysis = await self.analyze_context(
            f"Robot in {world_state.get('location', 'unknown location')} with objects {list(world_state.get('objects', {}).keys())}",
            goal
        )

        # Step 2: Decompose goal
        goal_decomp = await self.decompose_goal(goal, {
            "context_analysis": context_analysis.dict() if context_analysis else {},
            "world_state": world_state
        })

        if not goal_decomp:
            return {"success": False, "error": "Failed to decompose goal"}

        # Step 3: Generate action plans for each sub-goal
        action_plans = []
        for sub_goal in goal_decomp.sub_goals:
            action_plan = await self.generate_action_plan(sub_goal, world_state)
            if action_plan:
                action_plans.append(action_plan.dict())

        # Step 4: Create comprehensive plan
        comprehensive_plan = {
            "main_goal": goal,
            "decomposition": goal_decomp.dict(),
            "action_plans": action_plans,
            "context_analysis": context_analysis.dict() if context_analysis else {},
            "created_at": time.time()
        }

        return {"success": True, "plan": comprehensive_plan}

# Advanced planning utilities
class PlanningUtilities:
    """Utility functions for cognitive planning"""

    @staticmethod
    def calculate_plan_confidence(steps: List[Dict]) -> float:
        """Calculate overall confidence in a plan based on step confidences"""
        if not steps:
            return 0.0

        confidences = [step.get('confidence', 0.5) for step in steps]
        # Use geometric mean for more conservative estimate
        product = 1.0
        for conf in confidences:
            product *= conf
        return product ** (1.0 / len(confidences))

    @staticmethod
    def detect_conflicts_in_plan(plan: Dict[str, Any]) -> List[str]:
        """Detect potential conflicts in a plan"""
        conflicts = []
        steps = plan.get('action_plans', [])

        for i, step1 in enumerate(steps):
            for j, step2 in enumerate(steps[i+1:], i+1):
                # Check for resource conflicts
                step1_resources = step1.get('parameters', {}).get('arm', 'default')
                step2_resources = step2.get('parameters', {}).get('arm', 'default')

                if step1_resources == step2_resources and step1_resources != 'default':
                    conflicts.append(f"Resource conflict: steps {i} and {j} both need {step1_resources}")

        return conflicts

    @staticmethod
    def optimize_plan_order(plan: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize the order of plan steps"""
        # This is a simplified optimization
        # In practice, this would use more sophisticated algorithms
        optimized_plan = plan.copy()
        steps = optimized_plan.get('action_plans', [])

        # Sort by priority (if available) or by estimated duration
        # For now, just return the plan as is
        return optimized_plan

# Example of LLM integration
async def example_llm_integration():
    """Example of using LLM for cognitive planning"""
    # Initialize LLM interface (requires actual API key)
    llm_interface = LLMInterface(api_key="your-api-key-here")
    planner = LLMCognitivePlanner(llm_interface)

    # Example goal
    goal = "Bring a red cup from the kitchen table to the living room couch"

    # Example world state
    world_state = {
        "location": "kitchen",
        "objects": {
            "red_cup": {"location": "kitchen_table", "properties": {"color": "red", "graspable": True}},
            "kitchen_table": {"location": "kitchen_center", "properties": {"surface": True}},
            "living_room_couch": {"location": "living_room_center", "properties": {"furniture": True}}
        },
        "robot_state": {
            "location": "kitchen_entrance",
            "available_arms": ["left", "right"],
            "battery_level": 0.8
        }
    }

    # Create comprehensive plan
    result = await planner.create_comprehensive_plan(goal, world_state)

    if result["success"]:
        plan = result["plan"]
        print(f"Created plan for: {plan['main_goal']}")
        print(f"Decomposed into {len(plan['decomposition']['sub_goals'])} sub-goals")
        print(f"Generated {len(plan['action_plans'])} action plans")

        # Calculate plan confidence
        confidence = PlanningUtilities.calculate_plan_confidence(plan['action_plans'])
        print(f"Plan confidence: {confidence:.2f}")

        # Check for conflicts
        conflicts = PlanningUtilities.detect_conflicts_in_plan(plan)
        if conflicts:
            print(f"Detected conflicts: {conflicts}")
        else:
            print("No conflicts detected in plan")

        return plan
    else:
        print(f"Failed to create plan: {result['error']}")
        return None
```

## Planning Algorithms

Planning algorithms for LLM-based cognitive planning must balance the reasoning capabilities of language models with the practical constraints of robotic execution. The following implementation demonstrates various planning approaches.

```python
import heapq
from typing import List, Dict, Any, Tuple, Optional, Set
import networkx as nx
from dataclasses import dataclass
import numpy as np

@dataclass
class PlanNode:
    """Node in a planning graph"""
    state: Dict[str, Any]
    actions: List[str]
    cost: float
    heuristic: float
    parent: Optional['PlanNode'] = None

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

class HierarchicalTaskNetworkPlanner:
    """Hierarchical Task Network (HTN) planner using LLM guidance"""

    def __init__(self, llm_planner: LLMCognitivePlanner):
        self.llm_planner = llm_planner
        self.task_networks = {}  # Store predefined task networks

    def add_task_network(self, task_name: str, subtasks: List[Dict[str, Any]]):
        """Add a predefined task network"""
        self.task_networks[task_name] = subtasks

    async def plan_with_hierarchical_decomposition(self, goal: str, world_state: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """Plan using hierarchical decomposition guided by LLM"""
        # Use LLM to decompose the high-level goal
        goal_decomp = await self.llm_planner.decompose_goal(goal, world_state)

        if not goal_decomp:
            return None

        # Build hierarchical plan
        plan = []
        for i, sub_goal in enumerate(goal_decomp.sub_goals):
            # Generate action plan for each sub-goal
            action_plan = await self.llm_planner.generate_action_plan(sub_goal, world_state)

            if action_plan:
                # Add to overall plan
                for step in action_plan.steps:
                    step['sub_goal_idx'] = i
                    step['original_sub_goal'] = sub_goal
                    plan.append(step)

        return plan

class GraphBasedPlanner:
    """Graph-based planner for complex cognitive tasks"""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_state_transition(self, from_state: str, to_state: str, action: str, cost: float = 1.0):
        """Add a state transition to the planning graph"""
        self.graph.add_edge(from_state, to_state, action=action, cost=cost)

    def find_optimal_path(self, start_state: str, goal_state: str) -> Optional[List[Tuple[str, str, Dict]]]:
        """Find optimal path using Dijkstra's algorithm"""
        try:
            path = nx.shortest_path(self.graph, start_state, goal_state, weight='cost')
            edges = []
            for i in range(len(path) - 1):
                edge_data = self.graph[path[i]][path[i+1]]
                edges.append((path[i], path[i+1], edge_data))
            return edges
        except nx.NetworkXNoPath:
            return None

class AStarPlanner:
    """A* planner with LLM-heuristic guidance"""

    def __init__(self, llm_planner: LLMCognitivePlanner):
        self.llm_planner = llm_planner

    async def heuristic_estimate(self, current_state: Dict[str, Any], goal: str) -> float:
        """Use LLM to estimate heuristic distance to goal"""
        prompt = f"""
        Estimate how close the following state is to achieving the goal.
        State: {current_state}
        Goal: {goal}

        Return a number between 0 and 1, where:
        - 0 means the goal is achieved
        - 1 means very far from the goal
        - Numbers in between represent relative distance
        """

        response = await self.llm_planner.llm.generate_text(prompt, temperature=0.1)

        try:
            # Extract number from response
            match = re.search(r'(\d+\.?\d*)', response)
            if match:
                value = float(match.group(1))
                return min(1.0, max(0.0, value))  # Clamp between 0 and 1
        except:
            pass

        # Default fallback
        return 0.5

    async def plan(self, start_state: Dict[str, Any], goal: str,
                   max_steps: int = 100) -> Optional[List[Dict[str, Any]]]:
        """Plan using A* algorithm with LLM heuristic"""
        start_node = PlanNode(
            state=start_state,
            actions=[],
            cost=0.0,
            heuristic=await self.heuristic_estimate(start_state, goal)
        )

        open_set = [start_node]
        closed_set: Set[str] = set()

        step_count = 0
        while open_set and step_count < max_steps:
            current = heapq.heappop(open_set)
            state_key = str(current.state)  # Simple state keying

            if state_key in closed_set:
                continue

            closed_set.add(state_key)

            # Check if goal is reached (simplified)
            if await self._is_goal_reached(current.state, goal):
                return current.actions

            # Generate successor states
            successors = await self._generate_successors(current.state, current.actions)

            for next_state, action_taken in successors:
                next_state_key = str(next_state)
                if next_state_key in closed_set:
                    continue

                new_cost = current.cost + 1.0  # Uniform cost for simplicity
                heuristic = await self.heuristic_estimate(next_state, goal)

                next_node = PlanNode(
                    state=next_state,
                    actions=current.actions + [action_taken],
                    cost=new_cost,
                    heuristic=heuristic,
                    parent=current
                )

                heapq.heappush(open_set, next_node)

            step_count += 1

        return None  # No path found

    async def _is_goal_reached(self, state: Dict[str, Any], goal: str) -> bool:
        """Check if the goal has been reached"""
        # This would be more sophisticated in practice
        # For now, we'll use a simple check
        prompt = f"""
        Determine if the following state represents achievement of the goal.
        State: {state}
        Goal: {goal}

        Respond with 'true' if goal is achieved, 'false' otherwise.
        """

        response = await self.llm_planner.llm.generate_text(prompt, temperature=0.0)
        return 'true' in response.lower()

    async def _generate_successors(self, state: Dict[str, Any],
                                 current_actions: List[str]) -> List[Tuple[Dict[str, Any], str]]:
        """Generate successor states for planning"""
        # In a real system, this would interface with action execution
        # For simulation, we'll generate some possible state transitions
        successors = []

        # Simulate possible actions
        possible_actions = ["move_forward", "turn_left", "turn_right", "pick_up", "place"]

        for action in possible_actions:
            # Simulate state change based on action
            new_state = state.copy()
            new_state[f"last_action"] = action
            new_state[f"action_count"] = len(current_actions) + 1

            successors.append((new_state, action))

        return successors

class ReactivePlanner:
    """Reactive planner for handling unexpected situations"""

    def __init__(self):
        self.reactive_rules = []

    def add_reactive_rule(self, condition: Callable[[Dict], bool],
                         action: Callable[[Dict], Dict]):
        """Add a reactive rule: if condition, then action"""
        self.reactive_rules.append((condition, action))

    def execute_with_reactivity(self, plan: List[Dict[str, Any]],
                              initial_state: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Execute plan with reactive behavior"""
        current_state = initial_state.copy()
        executed_actions = []
        step = 0

        for action in plan:
            # Check reactive rules before executing each action
            for condition, rule_action in self.reactive_rules:
                if condition(current_state):
                    # Execute reactive action
                    current_state = rule_action(current_state)
                    executed_actions.append({
                        "type": "reactive",
                        "action": rule_action.__name__,
                        "step": step,
                        "state_before": current_state.copy()
                    })

            # Execute planned action
            # In a real system, this would interface with robot execution
            current_state["last_executed"] = action
            executed_actions.append({
                "type": "planned",
                "action": action,
                "step": step,
                "state_after": current_state.copy()
            })
            step += 1

        return executed_actions, current_state

class MultiModalPlanner:
    """Multi-modal planner combining symbolic and neural approaches"""

    def __init__(self, llm_planner: LLMCognitivePlanner):
        self.llm_planner = llm_planner
        self.symbolic_planner = GraphBasedPlanner()
        self.neural_components = {}  # Would include neural networks for perception, etc.

    async def create_hybrid_plan(self, goal: str, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Create a hybrid plan combining different planning approaches"""
        # Use LLM for high-level reasoning
        goal_analysis = await self.llm_planner.decompose_goal(goal, world_state)

        if not goal_analysis:
            return {"success": False, "error": "LLM goal analysis failed"}

        # Use graph-based planner for detailed path planning
        # This is a simplified example
        self.symbolic_planner.add_state_transition("start", "approach_object", "navigate", 1.0)
        self.symbolic_planner.add_state_transition("approach_object", "grasp_object", "pick_up", 1.0)
        self.symbolic_planner.add_state_transition("grasp_object", "move_to_destination", "navigate", 1.0)
        self.symbolic_planner.add_state_transition("move_to_destination", "place_object", "place", 1.0)

        path = self.symbolic_planner.find_optimal_path("start", "place_object")

        # Combine LLM analysis with symbolic plan
        hybrid_plan = {
            "llm_analysis": goal_analysis.dict(),
            "symbolic_path": path,
            "execution_strategy": "interleaved",
            "monitoring_plan": True
        }

        return {"success": True, "plan": hybrid_plan}

# Example of using different planning algorithms
async def example_planning_algorithms():
    """Example of using different planning algorithms"""
    # Initialize LLM interface
    llm_interface = LLMInterface(api_key="your-api-key-here")
    llm_planner = LLMCognitivePlanner(llm_interface)

    # Example goal and world state
    goal = "Navigate to kitchen, pick up a cup, and bring it to the table"
    world_state = {
        "robot_location": "living_room",
        "objects": {
            "cup": {"location": "kitchen_counter", "graspable": True},
            "kitchen_counter": {"location": "kitchen", "surface": True},
            "table": {"location": "dining_area", "surface": True}
        }
    }

    # HTN Planner
    htn_planner = HierarchicalTaskNetworkPlanner(llm_planner)
    htn_plan = await htn_planner.plan_with_hierarchical_decomposition(goal, world_state)
    print(f"HTN Plan: {len(htn_plan) if htn_plan else 0} steps")

    # A* Planner
    astar_planner = AStarPlanner(llm_planner)
    astar_plan = await astar_planner.plan(world_state, goal)
    print(f"A* Plan: {len(astar_plan) if astar_plan else 0} steps")

    # Multi-modal Planner
    multi_planner = MultiModalPlanner(llm_planner)
    multi_plan = await multi_planner.create_hybrid_plan(goal, world_state)
    print(f"Multi-modal plan created: {multi_plan['success']}")

    # Reactive planner example
    reactive_planner = ReactivePlanner()

    # Add a simple reactive rule
    def obstacle_detected(state):
        return state.get("obstacle_ahead", False)

    def avoid_obstacle(state):
        state["navigation_strategy"] = "avoidance_mode"
        state["obstacle_avoided"] = True
        return state

    reactive_planner.add_reactive_rule(obstacle_detected, avoid_obstacle)

    # Simulate execution with reactivity
    simple_plan = [{"action": "move_forward"}, {"action": "turn_right"}, {"action": "move_forward"}]
    initial_state = {"obstacle_ahead": True, "position": [0, 0]}
    executed, final_state = reactive_planner.execute_with_reactivity(simple_plan, initial_state)

    print(f"Reactive execution: {len(executed)} actions executed")
    print(f"Final state: {final_state}")

    return {
        "htn_plan": htn_plan,
        "astar_plan": astar_plan,
        "multi_plan": multi_plan,
        "reactive_execution": (executed, final_state)
    }
```

## Context Understanding

Context understanding is crucial for effective cognitive planning, as it allows the system to interpret goals and generate appropriate plans based on the current situation, environment, and constraints.

```python
from typing import Dict, List, Any, Optional, Set
import datetime
from dataclasses import dataclass
import spacy
from transformers import pipeline

@dataclass
class ContextFrame:
    """A frame representing context at a specific time"""
    timestamp: datetime.datetime
    spatial_context: Dict[str, Any]  # Location, objects, layout
    temporal_context: Dict[str, Any]  # Time of day, schedule, deadlines
    social_context: Dict[str, Any]  # People present, social norms, relationships
    task_context: Dict[str, Any]  # Current task, subtasks, progress
    resource_context: Dict[str, Any]  # Available resources, constraints, capabilities

class ContextExtractor:
    """Extract context from various sources"""

    def __init__(self):
        # Load NLP models
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None

        # Initialize other models
        self.entailment_model = pipeline(
            "text-classification",
            model="cross-encoder/nli-deberta-base"
        ) if False else None  # Disabled to avoid dependency issues

    def extract_from_text(self, text: str) -> Dict[str, Any]:
        """Extract context information from text"""
        if not self.nlp:
            return self._simple_text_extraction(text)

        doc = self.nlp(text)

        context = {
            "entities": [],
            "relations": [],
            "temporal_expressions": [],
            "spatial_expressions": [],
            "intent": self._extract_intent(doc)
        }

        # Extract named entities
        for ent in doc.ents:
            context["entities"].append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })

        # Extract temporal expressions
        for token in doc:
            if token.ent_type_ in ["TIME", "DATE"]:
                context["temporal_expressions"].append({
                    "text": token.text,
                    "type": token.ent_type_
                })

        # Extract spatial expressions
        spatial_keywords = ["kitchen", "living room", "bedroom", "bathroom", "table", "chair", "couch", "door", "window"]
        for token in doc:
            if token.lemma_.lower() in spatial_keywords:
                context["spatial_expressions"].append(token.text)

        return context

    def _simple_text_extraction(self, text: str) -> Dict[str, Any]:
        """Simple text extraction without NLP models"""
        import re

        # Extract time expressions
        time_patterns = [
            r'\d{1,2}:\d{2}',  # HH:MM
            r'(morning|afternoon|evening|night)',  # Time of day
            r'(today|tomorrow|yesterday|now)'  # Relative time
        ]

        times = []
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            times.extend(matches)

        # Extract spatial expressions
        spatial_keywords = ["kitchen", "living room", "bedroom", "bathroom", "table", "chair", "couch", "door", "window"]
        spaces = [word for word in text.lower().split() if word in spatial_keywords]

        # Extract potential objects
        object_keywords = ["cup", "bottle", "book", "phone", "keys", "water", "milk", "bread", "apple"]
        objects = [word for word in text.lower().split() if word in object_keywords]

        return {
            "entities": objects,
            "relations": [],
            "temporal_expressions": times,
            "spatial_expressions": spaces,
            "intent": self._simple_intent_extraction(text)
        }

    def _simple_intent_extraction(self, text: str) -> str:
        """Simple intent extraction from text"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["go", "move", "navigate", "walk", "come"]):
            return "navigation"
        elif any(word in text_lower for word in ["pick", "get", "take", "grasp", "hold"]):
            return "manipulation"
        elif any(word in text_lower for word in ["bring", "deliver", "carry", "transport"]):
            return "transportation"
        elif any(word in text_lower for word in ["find", "look", "search", "locate"]):
            return "search"
        else:
            return "unknown"

    def extract_from_world_state(self, world_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract context from world state"""
        context = {
            "spatial": {
                "robot_location": world_state.get("robot_location", "unknown"),
                "objects": list(world_state.get("objects", {}).keys()),
                "rooms": world_state.get("rooms", []),
                "navigation_map": world_state.get("navigation_map", {})
            },
            "temporal": {
                "current_time": world_state.get("current_time", time.time()),
                "battery_level": world_state.get("robot_state", {}).get("battery_level", 1.0),
                "last_action_time": world_state.get("robot_state", {}).get("last_action_time", 0)
            },
            "social": {
                "people_present": world_state.get("people", []),
                "interaction_history": world_state.get("interaction_history", [])
            },
            "task": {
                "current_task": world_state.get("current_task", {}),
                "task_progress": world_state.get("task_progress", 0.0),
                "subtasks": world_state.get("subtasks", [])
            }
        }
        return context

class ContextIntegrator:
    """Integrate multiple context sources"""

    def __init__(self):
        self.context_extractor = ContextExtractor()
        self.context_frames = []  # Store recent context frames
        self.max_context_history = 10

    def integrate_context(self, goal: str, world_state: Dict[str, Any],
                        user_input: str = None) -> ContextFrame:
        """Integrate context from multiple sources"""
        timestamp = datetime.datetime.now()

        # Extract context from different sources
        goal_context = self.context_extractor.extract_from_text(goal)
        world_context = self.context_extractor.extract_from_world_state(world_state)
        user_context = self.context_extractor.extract_from_text(user_input) if user_input else {}

        # Combine contexts
        spatial_context = {**world_context["spatial"], "goal_relevant_spaces": goal_context.get("spatial_expressions", [])}
        temporal_context = {**world_context["temporal"], "goal_temporal": goal_context.get("temporal_expressions", [])}
        social_context = world_context["social"]
        task_context = {**world_context["task"], "goal_intent": goal_context.get("intent", "unknown")}
        resource_context = {
            "capabilities": world_state.get("robot_state", {}).get("capabilities", []),
            "constraints": world_state.get("constraints", []),
            "available_objects": list(world_state.get("objects", {}).keys())
        }

        # Create context frame
        context_frame = ContextFrame(
            timestamp=timestamp,
            spatial_context=spatial_context,
            temporal_context=temporal_context,
            social_context=social_context,
            task_context=task_context,
            resource_context=resource_context
        )

        # Store in history
        self.context_frames.append(context_frame)
        if len(self.context_frames) > self.max_context_history:
            self.context_frames.pop(0)

        return context_frame

    def get_relevant_context(self, goal: str, time_window: float = 300) -> ContextFrame:
        """Get context relevant to a specific goal within a time window"""
        current_time = datetime.datetime.now()
        recent_frames = [
            frame for frame in self.context_frames
            if (current_time - frame.timestamp).total_seconds() < time_window
        ]

        if not recent_frames:
            # Return empty context frame if no recent frames
            return ContextFrame(
                timestamp=current_time,
                spatial_context={},
                temporal_context={},
                social_context={},
                task_context={},
                resource_context={}
            )

        # Merge context frames (simplified approach)
        merged_spatial = {}
        merged_temporal = {}
        merged_social = {}
        merged_task = {}
        merged_resources = {}

        for frame in recent_frames:
            merged_spatial.update(frame.spatial_context)
            merged_temporal.update(frame.temporal_context)
            merged_social.update(frame.social_context)
            merged_task.update(frame.task_context)
            merged_resources.update(frame.resource_context)

        return ContextFrame(
            timestamp=current_time,
            spatial_context=merged_spatial,
            temporal_context=merged_temporal,
            social_context=merged_social,
            task_context=merged_task,
            resource_context=merged_resources
        )

    def update_context_with_execution(self, action_result: Dict[str, Any]):
        """Update context based on action execution results"""
        current_frame = self.context_frames[-1] if self.context_frames else None
        if not current_frame:
            return

        # Update based on action result
        if action_result.get("success", False):
            # Update world state based on successful action
            action_type = action_result.get("action", {}).get("name", "")
            if action_type == "pick_up":
                obj_name = action_result.get("action", {}).get("parameters", {}).get("object_name")
                if obj_name and obj_name in current_frame.resource_context.get("available_objects", []):
                    current_frame.resource_context["available_objects"].remove(obj_name)
                    current_frame.resource_context["held_objects"] = current_frame.resource_context.get("held_objects", [])
                    current_frame.resource_context["held_objects"].append(obj_name)

        # Update timestamp
        current_frame.timestamp = datetime.datetime.now()

class ContextAwarePlanner:
    """Planning system that uses context information"""

    def __init__(self, llm_planner: LLMCognitivePlanner):
        self.llm_planner = llm_planner
        self.context_integrator = ContextIntegrator()

    async def plan_with_context(self, goal: str, world_state: Dict[str, Any],
                              user_input: str = None) -> Dict[str, Any]:
        """Plan using integrated context information"""
        # Integrate context
        context_frame = self.context_integrator.integrate_context(goal, world_state, user_input)

        # Create context-aware prompt
        context_description = self._describe_context(context_frame)

        prompt = f"""
        Plan to achieve the following goal considering the provided context:

        Goal: {goal}

        Context:
        {context_description}

        World State:
        {json.dumps(world_state, indent=2)}

        Generate a detailed plan that:
        1. Takes into account the current context
        2. Adapts to the environment and constraints
        3. Considers the social situation
        4. Uses available resources effectively

        Provide your plan in the standard format.
        """

        # Use LLM to generate context-aware plan
        response = await self.llm_planner.llm.generate_text(prompt)

        # Parse the response (simplified)
        plan = {
            "goal": goal,
            "context_aware_plan": response,
            "context_used": context_description,
            "timestamp": time.time()
        }

        return plan

    def _describe_context(self, context_frame: ContextFrame) -> str:
        """Create a textual description of the context"""
        description = f"""
        Spatial Context:
        - Robot Location: {context_frame.spatial_context.get('robot_location', 'unknown')}
        - Available Objects: {context_frame.spatial_context.get('objects', [])}
        - Rooms: {context_frame.spatial_context.get('rooms', [])}

        Temporal Context:
        - Current Time: {context_frame.temporal_context.get('current_time', 'unknown')}
        - Battery Level: {context_frame.temporal_context.get('battery_level', 'unknown')}

        Social Context:
        - People Present: {context_frame.social_context.get('people_present', [])}
        - Interaction History: {len(context_frame.social_context.get('interaction_history', []))} interactions

        Task Context:
        - Current Task: {context_frame.task_context.get('current_task', {})}
        - Task Progress: {context_frame.task_context.get('task_progress', 0.0)}

        Resource Context:
        - Capabilities: {context_frame.resource_context.get('capabilities', [])}
        - Constraints: {context_frame.resource_context.get('constraints', [])}
        - Available Objects: {context_frame.resource_context.get('available_objects', [])}
        """

        return description

    def adapt_plan_to_context(self, original_plan: Dict[str, Any],
                            new_context: ContextFrame) -> Dict[str, Any]:
        """Adapt an existing plan based on new context"""
        # This would involve more sophisticated adaptation logic
        # For now, we'll just return the original plan with context info
        adapted_plan = original_plan.copy()
        adapted_plan["adaptation_context"] = self._describe_context(new_context)
        adapted_plan["adaptation_timestamp"] = time.time()

        return adapted_plan

# Example of context understanding
async def example_context_understanding():
    """Example of context understanding in cognitive planning"""
    # Initialize systems
    llm_interface = LLMInterface(api_key="your-api-key-here")
    llm_planner = LLMCognitivePlanner(llm_interface)
    context_planner = ContextAwarePlanner(llm_planner)

    # Example scenario
    goal = "Bring coffee to John in the living room"
    world_state = {
        "robot_location": "kitchen",
        "objects": {
            "coffee": {"location": "kitchen_counter", "properties": {"hot": True, "graspable": True}},
            "living_room": {"location": "living_room", "properties": {"accessible": True}}
        },
        "people": ["John", "Mary"],
        "current_time": time.time(),
        "robot_state": {
            "battery_level": 0.7,
            "capabilities": ["navigation", "manipulation"],
            "location": "kitchen"
        }
    }
    user_input = "John is waiting in the living room for his coffee"

    # Plan with context
    plan = await context_planner.plan_with_context(goal, world_state, user_input)
    print(f"Context-aware plan generated")
    print(f"Context considered: {len(plan.get('context_aware_plan', ''))} characters of reasoning")

    # Simulate context change
    new_context = context_planner.context_integrator.get_relevant_context(goal)

    # Adapt plan to new context
    adapted_plan = context_planner.adapt_plan_to_context(plan, new_context)
    print(f"Plan adapted to new context")

    return {
        "original_plan": plan,
        "adapted_plan": adapted_plan
    }
```

## Task Decomposition

Task decomposition is a critical component of cognitive planning, breaking down complex goals into manageable subtasks that can be executed sequentially or in parallel.

```python
from typing import Dict, List, Any, Optional, Union
import asyncio
from dataclasses import dataclass
from enum import Enum
import networkx as nx

class TaskStatus(Enum):
    """Status of a task in the decomposition"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class TaskNode:
    """A node representing a task in the decomposition tree"""
    id: str
    description: str
    subtasks: List['TaskNode']
    dependencies: List[str]  # IDs of tasks that must be completed first
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0  # Lower number means higher priority
    estimated_duration: float = 1.0  # In seconds
    confidence: float = 0.5  # Confidence in successful completion
    resources_required: List[str] = None  # Resources needed for this task
    effects: List[str] = None  # Effects of completing this task

class TaskDecomposer:
    """System for decomposing tasks into subtasks using LLM guidance"""

    def __init__(self, llm_planner: LLMCognitivePlanner):
        self.llm_planner = llm_planner

    async def decompose_task(self, task_description: str,
                           context: Dict[str, Any] = None) -> TaskNode:
        """Decompose a task into subtasks using LLM"""
        prompt = f"""
        Decompose the following task into smaller, manageable subtasks:

        Task: {task_description}

        Context: {context or 'No additional context provided'}

        Consider:
        1. What are the logical steps required?
        2. Which steps depend on others?
        3. What resources are needed for each step?
        4. What are the expected outcomes of each step?
        5. How long might each step take?

        Provide the decomposition as a hierarchical structure with:
        - Task descriptions
        - Dependencies between tasks
        - Estimated duration for each task
        - Required resources
        - Expected effects of completing each task
        - Priority ranking (1-5, where 1 is highest priority)

        Format as JSON with structure:
        {{
            "task_id": "unique_id",
            "description": "task description",
            "subtasks": [subtask_objects],
            "dependencies": ["dependency_task_ids"],
            "estimated_duration": number_in_seconds,
            "resources_required": ["resource1", "resource2"],
            "effects": ["effect1", "effect2"],
            "priority": 1-5
        }}
        """

        response = await self.llm_planner.llm.generate_text(prompt)

        # Parse the response (simplified - in practice, use structured output)
        try:
            task_data = json.loads(response)
            return self._create_task_node(task_data)
        except json.JSONDecodeError:
            # If parsing fails, create a simple task node
            return TaskNode(
                id="task_0",
                description=task_description,
                subtasks=[],
                dependencies=[],
                priority=1,
                estimated_duration=1.0,
                confidence=0.5,
                resources_required=[],
                effects=[]
            )

    def _create_task_node(self, task_data: Dict[str, Any]) -> TaskNode:
        """Create a TaskNode from data dictionary"""
        subtasks = [self._create_task_node(sub) for sub in task_data.get('subtasks', [])]

        return TaskNode(
            id=task_data.get('task_id', f"task_{hash(task_data.get('description', '')) % 10000}"),
            description=task_data.get('description', ''),
            subtasks=subtasks,
            dependencies=task_data.get('dependencies', []),
            priority=task_data.get('priority', 3),
            estimated_duration=task_data.get('estimated_duration', 1.0),
            confidence=task_data.get('confidence', 0.5),
            resources_required=task_data.get('resources_required', []),
            effects=task_data.get('effects', [])
        )

    def build_task_graph(self, root_task: TaskNode) -> nx.DiGraph:
        """Build a dependency graph from the task decomposition"""
        graph = nx.DiGraph()

        def add_task_to_graph(task: TaskNode):
            graph.add_node(task.id, task=task, description=task.description)

            # Add dependency edges
            for dep_id in task.dependencies:
                graph.add_edge(dep_id, task.id, type='dependency')

            # Recursively add subtasks
            for subtask in task.subtasks:
                add_task_to_graph(subtask)
                graph.add_edge(task.id, subtask.id, type='subtask')

        add_task_to_graph(root_task)
        return graph

    def get_execution_order(self, task_graph: nx.DiGraph) -> List[str]:
        """Get the order in which tasks should be executed"""
        try:
            # Use topological sort to get execution order
            execution_order = list(nx.topological_sort(task_graph))
            return execution_order
        except nx.NetworkXUnfeasible:
            # If there are cycles, return an approximate order
            # In practice, you'd want to resolve the cycles first
            return list(task_graph.nodes())

    def prioritize_tasks(self, tasks: List[TaskNode],
                        resource_availability: Dict[str, int]) -> List[TaskNode]:
        """Prioritize tasks based on dependencies, resources, and urgency"""
        # Sort by priority (lower number = higher priority) and dependencies
        def task_priority(task: TaskNode):
            # Higher priority score for higher priority level
            priority_score = -task.priority

            # Adjust based on resource availability
            resource_penalty = 0
            for resource in task.resources_required or []:
                available = resource_availability.get(resource, 0)
                if available <= 0:
                    resource_penalty += 10  # Heavy penalty for unavailable resources

            # Adjust based on dependencies (tasks with fewer unmet dependencies get higher priority)
            # This is a simplified version
            return priority_score + resource_penalty

        return sorted(tasks, key=task_priority)

class HierarchicalTaskDecomposer(TaskDecomposer):
    """Advanced task decomposer with hierarchical capabilities"""

    def __init__(self, llm_planner: LLMCognitivePlanner):
        super().__init__(llm_planner)
        self.decomposition_cache = {}

    async def decompose_with_context(self, task_description: str,
                                   context_frame: ContextFrame) -> TaskNode:
        """Decompose task considering the current context"""
        context_info = self._extract_context_info(context_frame)

        prompt = f"""
        Decompose the following task considering the provided context:

        Task: {task_description}

        Context Information:
        - Spatial Context: {context_info['spatial']}
        - Temporal Context: {context_info['temporal']}
        - Social Context: {context_info['social']}
        - Resource Context: {context_info['resources']}

        Create a detailed decomposition that:
        1. Takes spatial layout into account
        2. Considers time constraints
        3. Respects social norms and people present
        4. Uses available resources efficiently
        5. Minimizes travel and maximizes efficiency

        Provide the decomposition in the same JSON format as before.
        """

        response = await self.llm_planner.llm.generate_text(prompt)

        try:
            task_data = json.loads(response)
            return self._create_task_node(task_data)
        except json.JSONDecodeError:
            # Fallback to simple decomposition
            return await self.decompose_task(task_description)

    def _extract_context_info(self, context_frame: ContextFrame) -> Dict[str, Any]:
        """Extract relevant information from context frame"""
        return {
            "spatial": {
                "robot_location": context_frame.spatial_context.get('robot_location'),
                "available_objects": context_frame.spatial_context.get('objects', []),
                "navigation_map": context_frame.spatial_context.get('navigation_map', {})
            },
            "temporal": {
                "current_time": context_frame.temporal_context.get('current_time'),
                "battery_level": context_frame.temporal_context.get('battery_level'),
                "time_deadline": context_frame.temporal_context.get('deadline', None)
            },
            "social": {
                "people_present": context_frame.social_context.get('people_present', []),
                "social_norms": context_frame.social_context.get('norms', [])
            },
            "resources": {
                "available_capabilities": context_frame.resource_context.get('capabilities', []),
                "resource_constraints": context_frame.resource_context.get('constraints', []),
                "available_objects": context_frame.resource_context.get('available_objects', [])
            }
        }

    async def decompose_with_learning(self, task_description: str,
                                    context_frame: ContextFrame) -> TaskNode:
        """Decompose task with learning from previous similar tasks"""
        # Check if we have similar tasks in cache
        cache_key = self._create_cache_key(task_description, context_frame)

        if cache_key in self.decomposition_cache:
            cached_decomp = self.decomposition_cache[cache_key]
            # Adapt cached decomposition to current context
            return await self._adapt_cached_decomposition(cached_decomp, context_frame)

        # If not in cache, decompose normally
        decomposition = await self.decompose_with_context(task_description, context_frame)

        # Store in cache
        self.decomposition_cache[cache_key] = decomposition

        return decomposition

    def _create_cache_key(self, task_description: str, context_frame: ContextFrame) -> str:
        """Create a cache key for the task decomposition"""
        import hashlib
        context_summary = (
            f"loc:{context_frame.spatial_context.get('robot_location', 'unknown')}_"
            f"obj:{len(context_frame.spatial_context.get('objects', []))}_"
            f"ppl:{len(context_frame.social_context.get('people_present', []))}"
        )
        full_key = f"{task_description}_{context_summary}"
        return hashlib.md5(full_key.encode()).hexdigest()

    async def _adapt_cached_decomposition(self, cached_decomp: TaskNode,
                                       context_frame: ContextFrame) -> TaskNode:
        """Adapt a cached decomposition to current context"""
        # This would involve more sophisticated adaptation
        # For now, we'll just return the cached decomposition
        return cached_decomp

class ParallelTaskScheduler:
    """Scheduler for executing tasks in parallel when possible"""

    def __init__(self):
        self.resource_manager = ResourceManager()

    def schedule_parallel_tasks(self, task_graph: nx.DiGraph,
                              available_resources: Dict[str, int]) -> List[List[str]]:
        """Schedule tasks for parallel execution"""
        execution_levels = []
        remaining_tasks = set(task_graph.nodes())

        while remaining_tasks:
            # Find tasks that can be executed at this level
            current_level = []

            for task_id in list(remaining_tasks):
                task_node = task_graph.nodes[task_id]['task']

                # Check if all dependencies are satisfied
                dependencies_met = all(
                    dep not in remaining_tasks
                    for dep in task_node.dependencies
                )

                # Check if resources are available
                resources_available = self.resource_manager.check_resources_available(
                    task_node.resources_required, available_resources
                )

                if dependencies_met and resources_available:
                    current_level.append(task_id)

            if not current_level:
                # No tasks can be scheduled, which might indicate a problem
                print(f"Warning: Could not schedule any tasks, remaining: {remaining_tasks}")
                break

            # Mark these tasks as scheduled
            for task_id in current_level:
                remaining_tasks.remove(task_id)

            execution_levels.append(current_level)

        return execution_levels

class ResourceManager:
    """Manage resources for task execution"""

    def __init__(self):
        self.resource_allocations = {}

    def check_resources_available(self, required_resources: List[str],
                               available_resources: Dict[str, int]) -> bool:
        """Check if required resources are available"""
        if not required_resources:
            return True

        for resource in required_resources:
            available = available_resources.get(resource, 0)
            allocated = self.resource_allocations.get(resource, 0)
            if (available - allocated) <= 0:
                return False

        return True

    def allocate_resources(self, task_node: TaskNode):
        """Allocate resources for a task"""
        for resource in task_node.resources_required or []:
            self.resource_allocations[resource] = self.resource_allocations.get(resource, 0) + 1

    def release_resources(self, task_node: TaskNode):
        """Release resources after task completion"""
        for resource in task_node.resources_required or []:
            current = self.resource_allocations.get(resource, 0)
            if current > 0:
                self.resource_allocations[resource] = current - 1

# Example of task decomposition
async def example_task_decomposition():
    """Example of task decomposition in cognitive planning"""
    # Initialize systems
    llm_interface = LLMInterface(api_key="your-api-key-here")
    llm_planner = LLMCognitivePlanner(llm_interface)

    # Create decomposer
    decomposer = HierarchicalTaskDecomposer(llm_planner)

    # Example task
    task = "Prepare a simple meal consisting of sandwich and drink, then serve to the person in the living room"

    # Create a context frame
    context_frame = ContextFrame(
        timestamp=datetime.datetime.now(),
        spatial_context={
            "robot_location": "kitchen",
            "objects": ["bread", "cheese", "ham", "bottle_water", "plate", "cup"],
            "rooms": ["kitchen", "living_room"]
        },
        temporal_context={
            "current_time": time.time(),
            "battery_level": 0.8
        },
        social_context={
            "people_present": ["John"],
            "location": "living_room"
        },
        task_context={},
        resource_context={
            "capabilities": ["navigation", "manipulation", "detection"],
            "available_objects": ["bread", "cheese", "ham", "bottle_water"]
        }
    )

    # Decompose task with context
    task_decomposition = await decomposer.decompose_with_context(task, context_frame)
    print(f"Task decomposed into {len(task_decomposition.subtasks) if task_decomposition.subtasks else 0} subtasks")

    # Build task graph
    task_graph = decomposer.build_task_graph(task_decomposition)
    print(f"Task dependency graph has {len(task_graph.nodes())} nodes and {len(task_graph.edges())} edges")

    # Get execution order
    execution_order = decomposer.get_execution_order(task_graph)
    print(f"Execution order: {execution_order[:5]}...")  # Show first 5 tasks

    # Schedule for parallel execution
    scheduler = ParallelTaskScheduler()
    available_resources = {"left_arm": 1, "right_arm": 1, "navigation_system": 1}
    execution_levels = scheduler.schedule_parallel_tasks(task_graph, available_resources)
    print(f"Tasks scheduled in {len(execution_levels)} parallel levels")

    for i, level in enumerate(execution_levels):
        print(f"  Level {i+1}: {len(level)} tasks")

    return {
        "decomposition": task_decomposition,
        "graph": task_graph,
        "execution_order": execution_order,
        "execution_levels": execution_levels
    }
```

## Execution Monitoring

Execution monitoring is critical for cognitive planning systems, allowing them to track progress, detect failures, and adapt plans in real-time.

```python
import asyncio
import time
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from enum import Enum
import threading
from datetime import datetime

class ExecutionStatus(Enum):
    """Status of plan execution"""
    NOT_STARTED = "not_started"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class MonitorEvent(Enum):
    """Types of monitoring events"""
    STEP_STARTED = "step_started"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    PLAN_ADAPTED = "plan_adapted"
    RESOURCE_CHANGED = "resource_changed"
    CONTEXT_CHANGED = "context_changed"
    GOAL_ACHIEVED = "goal_achieved"
    GOAL_FAILED = "goal_failed"

@dataclass
class ExecutionStep:
    """Represents a single step in plan execution"""
    id: str
    action: Dict[str, Any]
    status: ExecutionStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    confidence: float = 0.5

@dataclass
class ExecutionState:
    """Current state of plan execution"""
    status: ExecutionStatus
    current_step: int
    completed_steps: List[ExecutionStep]
    failed_steps: List[ExecutionStep]
    remaining_steps: List[Dict[str, Any]]
    progress: float
    start_time: datetime
    last_update: datetime

class ExecutionMonitor:
    """Monitor for plan execution with real-time feedback"""

    def __init__(self, llm_planner: LLMCognitivePlanner):
        self.llm_planner = llm_planner
        self.execution_state = None
        self.event_handlers = {}
        self.is_monitoring = False
        self.monitoring_thread = None
        self.lock = threading.Lock()

    def register_event_handler(self, event_type: MonitorEvent, handler: Callable):
        """Register a handler for specific monitoring events"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)

    async def trigger_event(self, event_type: MonitorEvent, data: Dict[str, Any]):
        """Trigger an event and notify all handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event_type, data)
                    else:
                        handler(event_type, data)
                except Exception as e:
                    print(f"Error in event handler: {e}")

    def start_monitoring(self, plan: List[Dict[str, Any]], initial_state: Dict[str, Any]):
        """Start monitoring execution of a plan"""
        with self.lock:
            self.execution_state = ExecutionState(
                status=ExecutionStatus.NOT_STARTED,
                current_step=0,
                completed_steps=[],
                failed_steps=[],
                remaining_steps=plan.copy(),
                progress=0.0,
                start_time=datetime.now(),
                last_update=datetime.now()
            )
            self.is_monitoring = True

    def stop_monitoring(self):
        """Stop monitoring execution"""
        with self.lock:
            self.is_monitoring = False
            if self.execution_state:
                self.execution_state.status = ExecutionStatus.CANCELLED

    async def monitor_step_execution(self, step_index: int, step: Dict[str, Any],
                                  execute_callback: Callable) -> ExecutionStep:
        """Monitor execution of a single step"""
        step_monitor = ExecutionStep(
            id=f"step_{step_index}",
            action=step,
            status=ExecutionStatus.NOT_STARTED
        )

        # Mark step as started
        step_monitor.status = ExecutionStatus.RUNNING
        step_monitor.start_time = datetime.now()

        await self.trigger_event(MonitorEvent.STEP_STARTED, {
            "step": step_monitor,
            "execution_state": self.execution_state
        })

        try:
            # Execute the step
            result = await execute_callback(step)

            # Mark as completed
            step_monitor.status = ExecutionStatus.COMPLETED
            step_monitor.end_time = datetime.now()
            step_monitor.result = result

            # Update execution state
            with self.lock:
                if self.execution_state:
                    self.execution_state.completed_steps.append(step_monitor)
                    self.execution_state.current_step += 1
                    self.execution_state.progress = len(self.execution_state.completed_steps) / len(self.execution_state.completed_steps + self.execution_state.failed_steps + self.execution_state.remaining_steps)
                    self.execution_state.last_update = datetime.now()

            await self.trigger_event(MonitorEvent.STEP_COMPLETED, {
                "step": step_monitor,
                "execution_state": self.execution_state
            })

        except Exception as e:
            # Mark as failed
            step_monitor.status = ExecutionStatus.FAILED
            step_monitor.end_time = datetime.now()
            step_monitor.error = str(e)

            # Update execution state
            with self.lock:
                if self.execution_state:
                    self.execution_state.failed_steps.append(step_monitor)
                    self.execution_state.current_step += 1
                    self.execution_state.last_update = datetime.now()

            await self.trigger_event(MonitorEvent.STEP_FAILED, {
                "step": step_monitor,
                "execution_state": self.execution_state,
                "error": str(e)
            })

        return step_monitor

    async def adapt_plan_during_execution(self, current_state: Dict[str, Any],
                                       failed_step: ExecutionStep) -> Optional[List[Dict[str, Any]]]:
        """Adapt the plan when a step fails"""
        prompt = f"""
        A step in the plan has failed. The failed step was:
        {failed_step.action}

        Error: {failed_step.error}

        Current world state: {current_state}

        Remaining steps in the plan: {self.execution_state.remaining_steps if self.execution_state else []}

        Suggest adaptations to the plan to handle this failure. Consider:
        1. Can the failed step be retried with different parameters?
        2. Should the step be skipped if possible?
        3. Does the failure require modifying subsequent steps?
        4. Is there an alternative approach to achieve the same goal?

        Provide an adapted plan in JSON format.
        """

        response = await self.llm_planner.llm.generate_text(prompt)

        try:
            adapted_plan = json.loads(response)
            await self.trigger_event(MonitorEvent.PLAN_ADAPTED, {
                "original_step": failed_step,
                "adaptation": adapted_plan,
                "execution_state": self.execution_state
            })
            return adapted_plan
        except json.JSONDecodeError:
            print(f"Failed to parse plan adaptation: {response}")
            return None

    def get_execution_progress(self) -> Dict[str, Any]:
        """Get current execution progress"""
        if not self.execution_state:
            return {"status": "no_execution_active"}

        with self.lock:
            return {
                "status": self.execution_state.status.value,
                "current_step": self.execution_state.current_step,
                "completed_steps": len(self.execution_state.completed_steps),
                "failed_steps": len(self.execution_state.failed_steps),
                "remaining_steps": len(self.execution_state.remaining_steps),
                "progress_percentage": self.execution_state.progress * 100,
                "execution_time": (datetime.now() - self.execution_state.start_time).total_seconds()
            }

    async def monitor_resource_usage(self, resource_callback: Callable):
        """Monitor resource usage during execution"""
        while self.is_monitoring:
            try:
                resource_status = await resource_callback()

                await self.trigger_event(MonitorEvent.RESOURCE_CHANGED, {
                    "resource_status": resource_status,
                    "execution_state": self.execution_state
                })

                # Check if resources are critically low
                if resource_status.get('battery_level', 1.0) < 0.1:
                    print("Warning: Battery level critically low")

                await asyncio.sleep(1)  # Check every second

            except Exception as e:
                print(f"Error monitoring resources: {e}")
                await asyncio.sleep(1)

class AdaptiveExecutionManager:
    """Manager for adaptive plan execution with monitoring"""

    def __init__(self, llm_planner: LLMCognitivePlanner):
        self.llm_planner = llm_planner
        self.monitor = ExecutionMonitor(llm_planner)
        self.context_integrator = ContextIntegrator()

    async def execute_plan_with_monitoring(self, plan: List[Dict[str, Any]],
                                        world_state: Dict[str, Any],
                                        execute_step_callback: Callable) -> Dict[str, Any]:
        """Execute a plan with full monitoring and adaptation capabilities"""
        # Start monitoring
        self.monitor.start_monitoring(plan, world_state)

        # Set up event handlers
        self.monitor.register_event_handler(MonitorEvent.STEP_FAILED, self._handle_step_failure)
        self.monitor.register_event_handler(MonitorEvent.CONTEXT_CHANGED, self._handle_context_change)

        results = []
        current_state = world_state.copy()

        for i, step in enumerate(plan):
            # Check if monitoring should continue
            if not self.monitor.is_monitoring:
                break

            # Monitor step execution
            step_result = await self.monitor.monitor_step_execution(i, step, execute_step_callback)
            results.append(step_result)

            # Update world state based on step result
            if step_result.status == ExecutionStatus.COMPLETED and step_result.result:
                current_state = self._update_world_state(current_state, step_result.result, step)

            # Check for context changes
            if i % 5 == 0:  # Check every 5 steps
                await self._check_context_changes(current_state)

        # Stop monitoring
        self.monitor.stop_monitoring()

        # Return execution summary
        return {
            "final_state": self.monitor.get_execution_progress(),
            "results": results,
            "success": all(r.status == ExecutionStatus.COMPLETED for r in results)
        }

    async def _handle_step_failure(self, event_type: MonitorEvent, data: Dict[str, Any]):
        """Handle when a step fails during execution"""
        failed_step = data['step']
        current_state = data.get('execution_state')

        print(f"Step {failed_step.id} failed: {failed_step.error}")

        # Try to adapt the plan
        adapted_plan = await self.monitor.adapt_plan_during_execution(current_state, failed_step)

        if adapted_plan:
            print(f"Plan adapted successfully")
        else:
            print(f"Could not adapt plan for failure")

    async def _handle_context_change(self, event_type: MonitorEvent, data: Dict[str, Any]):
        """Handle when context changes during execution"""
        context_change = data.get('context_change')
        print(f"Context changed: {context_change}")

    async def _check_context_changes(self, current_state: Dict[str, Any]):
        """Check for context changes that might affect execution"""
        # This would interface with perception systems in a real implementation
        # For simulation, we'll just return
        pass

    def _update_world_state(self, current_state: Dict[str, Any],
                           step_result: Dict[str, Any], action_taken: Dict[str, Any]) -> Dict[str, Any]:
        """Update world state based on step execution result"""
        new_state = current_state.copy()

        # Update based on action effects
        action_name = action_taken.get('name', '')
        if action_name == 'pick_up':
            obj_name = action_taken.get('parameters', {}).get('object_name')
            if obj_name and obj_name in new_state.get('objects', {}):
                del new_state['objects'][obj_name]
                new_state['held_objects'] = new_state.get('held_objects', [])
                new_state['held_objects'].append(obj_name)

        elif action_name == 'place':
            obj_name = action_taken.get('parameters', {}).get('object_name')
            target_location = action_taken.get('parameters', {}).get('target_location')
            if obj_name and obj_name in new_state.get('held_objects', []):
                new_state['held_objects'].remove(obj_name)
                new_state['objects'][obj_name] = {'location': target_location}

        # Update robot location if navigation occurred
        if action_name in ['move_to', 'navigate_to_object']:
            target = action_taken.get('parameters', {}).get('target_location')
            if target:
                new_state['robot_location'] = target

        return new_state

    async def execute_with_context_awareness(self, plan: List[Dict[str, Any]],
                                           initial_world_state: Dict[str, Any],
                                           execute_step_callback: Callable) -> Dict[str, Any]:
        """Execute plan with context awareness and adaptation"""
        # Integrate context
        context_frame = self.context_integrator.integrate_context(
            "execution_task", initial_world_state
        )

        # Execute with monitoring
        result = await self.execute_plan_with_monitoring(
            plan, initial_world_state, execute_step_callback
        )

        # Update context with execution results
        self.context_integrator.update_context_with_execution({
            "success": result["success"],
            "steps_completed": len([r for r in result["results"] if r.status == ExecutionStatus.COMPLETED]),
            "steps_failed": len([r for r in result["results"] if r.status == ExecutionStatus.FAILED])
        })

        return result

# Example execution monitoring
async def example_execution_monitoring():
    """Example of execution monitoring in cognitive planning"""
    # Initialize systems
    llm_interface = LLMInterface(api_key="your-api-key-here")
    llm_planner = LLMCognitivePlanner(llm_interface)
    execution_manager = AdaptiveExecutionManager(llm_planner)

    # Example plan (simplified)
    plan = [
        {"name": "navigate_to", "parameters": {"location": "kitchen"}, "type": "navigation"},
        {"name": "detect_object", "parameters": {"object": "cup"}, "type": "perception"},
        {"name": "pick_up", "parameters": {"object": "cup"}, "type": "manipulation"},
        {"name": "navigate_to", "parameters": {"location": "table"}, "type": "navigation"},
        {"name": "place", "parameters": {"object": "cup", "location": "table"}, "type": "manipulation"}
    ]

    # Initial world state
    world_state = {
        "robot_location": "start",
        "objects": {
            "cup": {"location": "kitchen_counter", "graspable": True},
            "kitchen_counter": {"location": "kitchen", "surface": True},
            "table": {"location": "dining_area", "surface": True}
        },
        "robot_state": {
            "battery_level": 0.8,
            "arm_status": "free"
        }
    }

    # Define step execution callback
    async def execute_step(step: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate step execution"""
        print(f"Executing: {step['name']} with {step.get('parameters', {})}")

        # Simulate different execution times based on action type
        action_times = {
            "navigation": 2.0,
            "perception": 1.0,
            "manipulation": 1.5
        }

        sleep_time = action_times.get(step.get('type', 'navigation'), 1.0)
        await asyncio.sleep(sleep_time * 0.1)  # Speed up for example

        # Simulate occasional failures
        import random
        if random.random() < 0.1:  # 10% failure rate
            raise Exception(f"Simulated failure in {step['name']}")

        return {"success": True, "result": f"completed_{step['name']}"}

    # Execute plan with monitoring
    result = await execution_manager.execute_with_context_awareness(
        plan, world_state, execute_step
    )

    print(f"Execution completed with success: {result['success']}")
    print(f"Steps completed: {len([r for r in result['results'] if r.status == ExecutionStatus.COMPLETED])}")
    print(f"Steps failed: {len([r for r in result['results'] if r.status == ExecutionStatus.FAILED])}")

    return result

# Run the example
async def run_all_examples():
    """Run all examples to demonstrate the cognitive planning system"""
    print("=== LLM Cognitive Planning Examples ===\n")

    # Note: These examples require an actual OpenAI API key to run properly
    # The code structure demonstrates the implementation approach

    print("1. Basic Cognitive Planning:")
    try:
        # result1 = await example_cognitive_planning()
        print("  - Cognitive planning structure demonstrated")
    except Exception as e:
        print(f"  - Example skipped due to API requirements: {e}")

    print("\n2. LLM Integration:")
    try:
        # result2 = await example_llm_integration()
        print("  - LLM integration structure demonstrated")
    except Exception as e:
        print(f"  - Example skipped due to API requirements: {e}")

    print("\n3. Planning Algorithms:")
    try:
        # result3 = await example_planning_algorithms()
        print("  - Planning algorithms structure demonstrated")
    except Exception as e:
        print(f"  - Example skipped due to API requirements: {e}")

    print("\n4. Context Understanding:")
    try:
        # result4 = await example_context_understanding()
        print("  - Context understanding structure demonstrated")
    except Exception as e:
        print(f"  - Example skipped due to API requirements: {e}")

    print("\n5. Task Decomposition:")
    try:
        # result5 = await example_task_decomposition()
        print("  - Task decomposition structure demonstrated")
    except Exception as e:
        print(f"  - Example skipped due to API requirements: {e}")

    print("\n6. Execution Monitoring:")
    try:
        # result6 = await example_execution_monitoring()
        print("  - Execution monitoring structure demonstrated")
    except Exception as e:
        print(f"  - Example skipped due to API requirements: {e}")

    print("\n=== All examples completed ===")

if __name__ == "__main__":
    asyncio.run(run_all_examples())
```

## Summary

LLM-based cognitive planning enables humanoid robots to perform complex, multi-step tasks by leveraging the reasoning capabilities of large language models. The implementation includes:

1. **Cognitive Planning Framework**: A comprehensive system that understands goals, analyzes world state, generates plans, and monitors execution with adaptation capabilities.

2. **LLM Integration**: Structured integration of large language models for goal understanding, plan generation, and context analysis with proper response parsing and validation.

3. **Planning Algorithms**: Multiple planning approaches including hierarchical task networks, graph-based planning, A* search with LLM heuristics, and reactive planning for handling unexpected situations.

4. **Context Understanding**: Advanced context extraction and integration from multiple sources including text, world state, temporal factors, and social context.

5. **Task Decomposition**: Hierarchical task decomposition with dependency management, resource allocation, and parallel execution scheduling.

6. **Execution Monitoring**: Real-time monitoring with event handling, failure adaptation, and context-aware plan adjustment during execution.

The system is designed to be robust and adaptive, capable of handling ambiguous natural language commands and adjusting to changing environmental conditions. The modular architecture allows for integration with different LLM providers and robotic platforms, making it suitable for a wide range of humanoid robot applications requiring sophisticated cognitive capabilities.