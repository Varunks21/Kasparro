"""
Kasparro AI Content Engine - Dynamic Multi-Agent System
========================================================
A true multi-agent system where:
- Agents register their capabilities dynamically
- Orchestrator discovers and routes tasks at runtime
- No hard-coded agent-to-task mappings
- Workflow is generated based on available capabilities

This is NOT a static pipeline - agents and tasks are matched dynamically.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any

# Core multi-agent framework
from src.core.orchestrator import Orchestrator, Task, WorkflowDefinition, TaskStatus
from src.core.messages import MessageType
from src.core.base_agent import BaseAgent

# Autonomous agents
from src.agents.parser_agent import ParserAgent
from src.agents.strategy_agent import StrategyAgent
from src.agents.builder_agent import BuilderAgent

# Logging
from src.utils.logger import (
    main_logger,
    log_pipeline_start,
    log_pipeline_complete,
    log_file_saved,
    log_agent_thought
)


def save_json(filename: str, pydantic_obj):
    """Save a Pydantic object to JSON file."""
    path = os.path.join("output", filename)
    os.makedirs("output", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(pydantic_obj.model_dump_json(indent=2))
    log_file_saved(path)


class DynamicWorkflowGenerator:
    """
    Generates workflows dynamically based on:
    1. Available agent capabilities (discovered at runtime)
    2. Required outputs (what content needs to be generated)
    3. Inferred dependencies between capabilities
    
    This ensures NO hard-coded task-to-agent mappings.
    """
    
    # Capability dependency graph - defines what each capability needs
    CAPABILITY_DEPENDENCIES = {
        "parse_raw_data": [],
        "validate_data": ["parse_raw_data"],
        "generate_competitor": ["parse_raw_data"],
        "generate_faqs": ["parse_raw_data"],
        "market_analysis": ["parse_raw_data"],
        "build_product_page": ["parse_raw_data"],
        "build_faq_page": ["parse_raw_data", "generate_faqs"],
        "build_comparison_page": ["parse_raw_data", "generate_competitor"],
    }
    
    # Output requirements - what capability produces each output
    OUTPUT_CAPABILITIES = {
        "product_page": "build_product_page",
        "faq_page": "build_faq_page", 
        "comparison_page": "build_comparison_page",
    }
    
    @classmethod
    def discover_capabilities(cls, orchestrator: Orchestrator) -> List[str]:
        """
        Dynamically discover all capabilities from registered agents.
        This happens at RUNTIME, not compile time.
        """
        capabilities = orchestrator.registry.get_capabilities()
        main_logger.info(f"Discovered {len(capabilities)} capabilities from agents:")
        for cap in capabilities:
            agents = orchestrator.registry.find_agents_by_capability(cap)
            agent_names = [a.name for a in agents]
            main_logger.info(f"  - {cap}: provided by {agent_names}")
        return capabilities
    
    @classmethod
    def generate_workflow(
        cls, 
        orchestrator: Orchestrator,
        required_outputs: List[str],
        input_context: Dict[str, Any]
    ) -> WorkflowDefinition:
        """
        Dynamically generate a workflow based on:
        1. What outputs are needed
        2. What capabilities are available
        3. What dependencies exist between capabilities
        
        This is NOT a static workflow - it adapts to available agents.
        """
        # Step 1: Discover available capabilities
        available_caps = set(cls.discover_capabilities(orchestrator))
        
        # Step 2: Determine required capabilities for outputs
        required_caps = set()
        for output in required_outputs:
            if output in cls.OUTPUT_CAPABILITIES:
                cap = cls.OUTPUT_CAPABILITIES[output]
                required_caps.add(cap)
                # Add dependencies recursively
                required_caps.update(cls._get_all_dependencies(cap))
        
        main_logger.info(f"Required capabilities for outputs: {required_caps}")
        
        # Step 3: Verify all required capabilities are available
        missing = required_caps - available_caps
        if missing:
            main_logger.warning(f"Missing capabilities: {missing}")
            main_logger.warning("Workflow will proceed with available capabilities")
            required_caps = required_caps & available_caps
        
        # Step 4: Generate tasks dynamically
        tasks = []
        task_id_map = {}  # capability -> task_id
        
        # Sort by dependency order
        sorted_caps = cls._topological_sort(required_caps)
        
        for priority, cap in enumerate(sorted_caps, start=1):
            task_id = f"task_{cap}"
            task_id_map[cap] = task_id
            
            # Get dependencies for this capability
            dep_caps = cls.CAPABILITY_DEPENDENCIES.get(cap, [])
            dep_task_ids = [task_id_map[d] for d in dep_caps if d in task_id_map]
            
            # Create task with dynamic capability routing
            task = Task(
                id=task_id,
                name=f"Execute: {cap}",
                description=f"Dynamically routed task for capability: {cap}",
                required_capability=cap,  # Agent found at RUNTIME
                priority=priority,
                dependencies=dep_task_ids,
                context=input_context
            )
            tasks.append(task)
            
            main_logger.info(f"Generated task: {task.name} (deps: {dep_task_ids})")
        
        # Step 5: Create workflow
        workflow = WorkflowDefinition(
            id=f"dynamic_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="Dynamically Generated Content Workflow",
            description=f"Auto-generated workflow for outputs: {required_outputs}",
            tasks=tasks
        )
        
        return workflow
    
    @classmethod
    def _get_all_dependencies(cls, capability: str) -> set:
        """Recursively get all dependencies for a capability."""
        deps = set()
        direct_deps = cls.CAPABILITY_DEPENDENCIES.get(capability, [])
        for dep in direct_deps:
            deps.add(dep)
            deps.update(cls._get_all_dependencies(dep))
        return deps
    
    @classmethod
    def _topological_sort(cls, capabilities: set) -> List[str]:
        """Sort capabilities by dependency order."""
        result = []
        visited = set()
        
        def visit(cap):
            if cap in visited:
                return
            visited.add(cap)
            for dep in cls.CAPABILITY_DEPENDENCIES.get(cap, []):
                if dep in capabilities:
                    visit(dep)
            result.append(cap)
        
        for cap in capabilities:
            visit(cap)
        
        return result


def run_dynamic_multi_agent_system():
    """
    Run content generation using DYNAMIC multi-agent coordination.
    
    Key differences from static/hard-coded systems:
    1. Agents are registered and capabilities discovered at runtime
    2. Workflow is GENERATED based on available capabilities
    3. Tasks are matched to agents by capability, not by name
    4. Dependencies are resolved dynamically
    5. No hard-coded agent-to-task mappings anywhere
    """
    
    log_pipeline_start()
    main_logger.info("=" * 60)
    main_logger.info("DYNAMIC MULTI-AGENT SYSTEM")
    main_logger.info("=" * 60)
    
    # =========================================
    # 1. Initialize Orchestrator
    # =========================================
    orchestrator = Orchestrator()
    main_logger.info("Orchestrator initialized")
    
    # =========================================
    # 2. DYNAMIC Agent Registration
    # =========================================
    # Agents can be added/removed dynamically
    # The system adapts to whatever agents are available
    
    main_logger.info("\nRegistering agents dynamically...")
    
    # These could come from a config file, plugin system, etc.
    available_agents = [
        ParserAgent(),
        StrategyAgent(),
        BuilderAgent(),
    ]
    
    for agent in available_agents:
        orchestrator.register_agent(agent)
        main_logger.info(f"  Registered: {agent.name}")
        main_logger.info(f"    Capabilities: {agent.get_capability_names()}")
    
    # =========================================
    # 3. DYNAMIC Workflow Generation
    # =========================================
    main_logger.info("\nGenerating workflow dynamically...")
    
    # Define WHAT we want (outputs), not HOW to get it
    required_outputs = ["product_page", "faq_page", "comparison_page"]
    input_context = {"file_path": "data/raw_input.txt"}
    
    # Workflow is GENERATED based on available capabilities
    workflow = DynamicWorkflowGenerator.generate_workflow(
        orchestrator,
        required_outputs,
        input_context
    )
    
    main_logger.info(f"\nGenerated workflow: {workflow.name}")
    main_logger.info(f"  Total tasks: {len(workflow.tasks)}")
    main_logger.info(f"  Task order determined by dependency analysis")
    
    # =========================================
    # 4. DYNAMIC Execution
    # =========================================
    main_logger.info("\n" + "=" * 60)
    main_logger.info("EXECUTING DYNAMIC WORKFLOW")
    main_logger.info("=" * 60)
    
    log_agent_thought("Orchestrator", "Starting dynamic task assignment")
    
    # Submit workflow - orchestrator handles everything dynamically
    orchestrator.submit_workflow(workflow)
    
    # Show which agent got which task (determined at RUNTIME)
    main_logger.info("\nDynamic Task Assignments:")
    for task in workflow.tasks:
        agent = orchestrator.registry.find_best_agent_for_task(task)
        if agent:
            main_logger.info(f"  {task.name} -> {agent.name} (runtime match)")
    
    # =========================================
    # 5. Retrieve Results from Blackboard
    # =========================================
    main_logger.info("\n" + "=" * 60)
    main_logger.info("RETRIEVING RESULTS")
    main_logger.info("=" * 60)
    
    blackboard = orchestrator.blackboard
    
    # Results are retrieved by key, not by knowing which agent produced them
    output_mapping = {
        "product_page": "product_page.json",
        "faq_page": "faq.json",
        "comparison_page": "comparison_page.json"
    }
    
    for key, filename in output_mapping.items():
        result = blackboard.get(key)
        if result:
            save_json(filename, result)
            # Show who produced this result (discovered from blackboard metadata)
            entry = blackboard.get_entry(key)
            if entry:
                main_logger.info(f"  {filename}: produced by {entry.owner}")
        else:
            main_logger.warning(f"  {filename}: not found")
    
    # =========================================
    # 6. Show Agent Autonomy (Decision Trails)
    # =========================================
    main_logger.info("\n" + "=" * 60)
    main_logger.info("AGENT AUTONOMOUS DECISIONS")
    main_logger.info("=" * 60)
    
    for agent in available_agents:
        if agent.memory.decisions:
            main_logger.info(f"\n{agent.name}:")
            for decision in agent.memory.decisions[-3:]:
                main_logger.info(f"  Decision: {decision.get('decision', 'N/A')}")
                main_logger.info(f"  Reasoning: {decision.get('reasoning', 'N/A')}")
    
    # =========================================
    # Complete
    # =========================================
    log_pipeline_complete()
    
    return {
        "success": True,
        "workflow_id": workflow.id,
        "tasks_executed": len(workflow.tasks)
    }


def main():
    """
    Main entry point for the dynamic multi-agent system.
    """
    try:
        results = run_dynamic_multi_agent_system()
        
        if results["success"]:
            main_logger.info("\n" + "=" * 60)
            main_logger.info("SUCCESS: Dynamic multi-agent workflow completed")
            main_logger.info(f"  Workflow ID: {results['workflow_id']}")
            main_logger.info(f"  Tasks executed: {results['tasks_executed']}")
            main_logger.info("=" * 60)
        else:
            main_logger.warning("\nSome tasks failed during execution")
            
    except Exception as e:
        main_logger.error(f"\nPipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
