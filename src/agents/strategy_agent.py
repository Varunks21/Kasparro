"""
Strategy Agent - Autonomous Strategic Planning Agent
====================================================
An autonomous agent responsible for strategic tasks like
competitor analysis and FAQ generation.
"""

import typing
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.core.base_agent import BaseAgent, AgentCapability, AgentGoal, AgentState
from src.core.messages import Message, MessageType
from src.models.internal import ProductData, CompetitorData
from src.utils.llm_client import get_structured_data
from src.utils.logger import get_agent_logger

logger = get_agent_logger("StrategyAgent")


# Internal model for FAQ questions
class QuestionList(BaseModel):
    questions: typing.List[str]


class StrategyAgent(BaseAgent):
    """
    Autonomous agent for strategic planning and analysis.
    
    Capabilities:
    - generate_competitor: Create fictional competitor for comparison
    - generate_faqs: Generate FAQ questions based on product data
    - market_analysis: Analyze market positioning
    
    This agent makes autonomous decisions about competitive
    positioning and customer experience strategy.
    """
    
    def __init__(self):
        super().__init__(
            agent_id="strategy_agent",
            name="Strategy Agent",
            description="Autonomous agent for strategic analysis and planning",
            capabilities=[
                AgentCapability(
                    name="generate_competitor",
                    description="Generate fictional competitor data for comparison",
                    input_types=["ProductData"],
                    output_types=["CompetitorData"]
                ),
                AgentCapability(
                    name="generate_faqs",
                    description="Generate FAQ questions answerable from product data",
                    input_types=["ProductData"],
                    output_types=["List[str]"]
                ),
                AgentCapability(
                    name="market_analysis",
                    description="Analyze market positioning and strategy",
                    input_types=["ProductData"],
                    output_types=["MarketAnalysis"]
                )
            ]
        )
        
    def plan(self, goal: AgentGoal) -> Optional[List[Dict[str, Any]]]:
        """
        Create a strategic plan based on the goal.
        
        The agent autonomously determines what strategic
        tasks need to be performed.
        """
        self.logger.info(f"Planning strategy for goal: {goal.description}")
        
        context = goal.context
        plan = []
        
        # Step 1: Acquire product data (from blackboard or context)
        plan.append({
            "action": "acquire_product_data",
            "params": {}
        })
        
        # Decision: What strategic tasks to perform?
        task_type = context.get("task_type", "full_strategy")
        
        if task_type == "competitor" or task_type == "full_strategy":
            self.memory.record_decision(
                "Generate competitor analysis",
                "Competitor data needed for comparison page"
            )
            plan.append({
                "action": "generate_competitor",
                "params": {}
            })
            
        if task_type == "faqs" or task_type == "full_strategy":
            self.memory.record_decision(
                "Generate FAQ questions",
                "FAQ questions needed for FAQ page"
            )
            plan.append({
                "action": "generate_faqs",
                "params": {}
            })
            
        # Step: Publish all results
        plan.append({
            "action": "publish_results",
            "params": {"goal_id": goal.id}
        })
        
        self.logger.info(f"Strategy plan created with {len(plan)} steps")
        return plan
        
    def execute(self, plan: List[Dict[str, Any]], goal: AgentGoal) -> bool:
        """Execute the strategic plan."""
        
        product_data = None
        competitor_data = None
        faq_questions = None
        
        for step in plan:
            action = step["action"]
            params = step["params"]
            
            self.logger.debug(f"Executing strategy step: {action}")
            
            try:
                if action == "acquire_product_data":
                    # Try to get from blackboard first
                    product_data = self.read_from_blackboard("product_data")
                    
                    if not product_data:
                        # Try from context
                        if "product_data" in goal.context:
                            product_data = ProductData(**goal.context["product_data"])
                            
                    if not product_data:
                        self.logger.error("No product data available")
                        self.memory.record_outcome(action, False, "Product data not found")
                        
                        # Request from parser agent
                        self.request_assistance(
                            "Need product data for strategy",
                            "parse_raw_data"
                        )
                        return False
                        
                    self.memory.record_outcome(action, True, f"Got data for: {product_data.name}")
                    
                elif action == "generate_competitor":
                    competitor_data = self._generate_competitor(product_data)
                    if not competitor_data:
                        self.memory.record_outcome(action, False, "Failed to generate competitor")
                        return False
                    self.memory.record_outcome(action, True, f"Generated: {competitor_data.name}")
                    
                elif action == "generate_faqs":
                    faq_questions = self._generate_faqs(product_data)
                    if not faq_questions:
                        self.memory.record_outcome(action, False, "Failed to generate FAQs")
                        return False
                    self.memory.record_outcome(action, True, f"Generated {len(faq_questions)} questions")
                    
                elif action == "publish_results":
                    self._publish_results(competitor_data, faq_questions, params.get("goal_id"))
                    self.memory.record_outcome(action, True)
                    
            except Exception as e:
                self.logger.error(f"Strategy step '{action}' failed: {e}")
                self.memory.record_outcome(action, False, str(e))
                return False
                
        return True
        
    def _generate_competitor(self, product: ProductData) -> Optional[CompetitorData]:
        """
        Generate a fictional competitor product.
        
        The agent uses its strategic knowledge to create a
        realistic but inferior competitor for comparison.
        Falls back to rule-based generation if LLM fails.
        """
        self.logger.info(f"Generating competitor for: {product.name}")
        
        self.memory.record_observation({
            "type": "competitor_generation_start",
            "base_product": product.name,
            "price_point": product.price_inr
        })
        
        prompt = f"""
        You are a Strategic Competitor Analysis Agent.
        
        Task: Create a FICTIONAL competitor product ("Product B") to compare against our product.
        
        Our Product Details:
        - Name: {product.name}
        - Price: ₹{product.price_inr}
        - Concentration: {product.concentration}
        - Ingredients: {', '.join(product.key_ingredients)}
        - Benefits: {', '.join(product.benefits)}
        
        STRICT Rules for Competitor:
        1. Name: Invent a realistic sounding skincare brand name (NOT a real brand).
        2. Price: Make it 15-25% cheaper than our product (₹{int(product.price_inr * 0.75)} to ₹{int(product.price_inr * 0.85)}).
        3. Ingredients: Use 3-4 common but LESS effective alternatives:
           - Lower concentration of active ingredients
           - No stabilizing ingredients
           - Missing key synergistic compounds
        4. Pros: List exactly 2 genuine-sounding pros (e.g., "Affordable", "Gentle formula")
        5. Cons: List exactly 2 specific cons that highlight our product's advantages (e.g., "Lower active concentration", "Missing key antioxidants")
        
        Return the result strictly as the CompetitorData JSON structure.
        """
        
        try:
            result = get_structured_data(prompt, CompetitorData)
            
            self.logger.info(f"[OK] Competitor generated via LLM: {result.name}")
            self.logger.debug(f"  - Price: Rs.{result.price_inr}")
            self.logger.debug(f"  - Pros: {result.pros}")
            self.logger.debug(f"  - Cons: {result.cons}")
            
            return result
            
        except Exception as e:
            self.logger.warning(f"LLM competitor generation failed, using rule-based fallback: {e}")
            return self._generate_competitor_fallback(product)
            
    def _generate_competitor_fallback(self, product: ProductData) -> CompetitorData:
        """
        Rule-based fallback for competitor generation when LLM is unavailable.
        """
        self.logger.info("Using rule-based competitor generation")
        
        # Generate competitor price (20% cheaper)
        competitor_price = int(product.price_inr * 0.80)
        
        # Generate generic competitor ingredients (less effective alternatives)
        generic_ingredients = ["Ascorbic Acid (5%)", "Glycerin", "Aloe Vera Extract"]
        
        competitor = CompetitorData(
            name="BudgetGlow Basic Serum",
            price_inr=competitor_price,
            key_ingredients=generic_ingredients,
            pros=["Budget-friendly price", "Gentle formula suitable for beginners"],
            cons=["Lower active concentration (5% vs 10%)", "Missing stabilizing antioxidants"]
        )
        
        self.logger.info(f"[OK] Competitor generated via fallback: {competitor.name}")
        return competitor
            
    def _generate_faqs(self, product: ProductData) -> Optional[List[str]]:
        """
        Generate FAQ questions that can be answered from product data.
        
        The agent strategically creates questions that showcase
        the product's strengths and address customer concerns.
        Falls back to template-based questions if LLM fails.
        """
        self.logger.info(f"Generating FAQ questions for: {product.name}")
        
        self.memory.record_observation({
            "type": "faq_generation_start",
            "product": product.name,
            "data_points": len(product.key_ingredients) + len(product.benefits)
        })
        
        # Build comprehensive product context
        product_context = f"""
        Product Name: {product.name}
        Concentration: {product.concentration}
        Price: ₹{product.price_inr}
        Suitable Skin Types: {', '.join(product.skin_type)}
        
        Key Ingredients: {', '.join(product.key_ingredients)}
        
        Benefits: {', '.join(product.benefits)}
        
        Usage Instructions: {product.usage_instructions}
        
        Side Effects/Warnings: {product.side_effects or 'None specified'}
        """
        
        prompt = f"""
        You are a Customer Experience Strategist creating a comprehensive FAQ section.
        
        IMPORTANT: Generate questions that CAN BE 100% ACCURATELY ANSWERED using ONLY the product data below.
        Do NOT generate questions about information not present in the data.
        
        PRODUCT DATA:
        {product_context}
        
        Generate exactly 15 distinct questions across these categories:
        
        1. INFORMATIONAL QUESTIONS (3 questions) - General product information
           Examples: "What is this product?", "What are the key ingredients?", "What concentration is used?"
        
        2. USAGE QUESTIONS (3 questions) - About how to use the product
           Examples: "How much should I apply?", "When should I use it?", "How often should beginners use it?"
        
        3. SAFETY QUESTIONS (3 questions) - About warnings and side effects
           Examples: "What side effects might occur?", "Is it safe during pregnancy?", "Do I need sunscreen?"
        
        4. PURCHASE QUESTIONS (2 questions) - About buying and value
           Examples: "What is the price?", "Is it worth the price?", "What value does it offer?"
        
        5. COMPARISON QUESTIONS (2 questions) - Comparing to alternatives
           Examples: "How does this compare to other products?", "What makes this product different?"
        
        6. SKIN TYPE QUESTIONS (2 questions) - About suitability
           Examples: "What skin types is this for?", "Can sensitive skin use this?", "Is it good for oily skin?"
        
        CRITICAL: Every question MUST be answerable from the product data provided above.
        
        Output: A list of exactly 15 question strings.
        """
        
        try:
            result = get_structured_data(prompt, QuestionList)
            
            self.logger.info(f"[OK] Generated {len(result.questions)} FAQ questions via LLM")
            for i, q in enumerate(result.questions[:3], 1):
                self.logger.debug(f"  - Q{i}: {q[:50]}...")
            
            return result.questions
            
        except Exception as e:
            self.logger.warning(f"LLM FAQ generation failed, using template fallback: {e}")
            return self._generate_faqs_fallback(product)
            
    def _generate_faqs_fallback(self, product: ProductData) -> List[str]:
        """
        Template-based fallback for FAQ generation when LLM is unavailable.
        Generates 15 product-specific questions using templates.
        """
        self.logger.info("Using template-based FAQ generation")
        
        questions = [
            # INFORMATIONAL (3)
            f"What is {product.name}?",
            f"What are the key ingredients in {product.name}?",
            f"What concentration of active ingredients does {product.name} contain?",
            # USAGE (3)
            f"How do I use {product.name}?",
            f"How often should I apply {product.name}?",
            f"When is the best time to use {product.name}?",
            # SAFETY (3)
            f"What are the potential side effects of {product.name}?",
            f"Is {product.name} safe for sensitive skin?",
            f"Do I need to use sunscreen with {product.name}?",
            # PURCHASE (2)
            f"What is the price of {product.name}?",
            f"Is {product.name} worth the investment?",
            # COMPARISON (2)
            f"What makes {product.name} different from other products?",
            f"How does {product.name} compare to budget alternatives?",
            # SKIN TYPE (2)
            f"What skin types is {product.name} suitable for?",
            f"Can I use {product.name} if I have oily skin?",
        ]
        
        self.logger.info(f"[OK] Generated {len(questions)} FAQ questions via fallback templates")
        return questions
            
    def _publish_results(
        self,
        competitor_data: Optional[CompetitorData],
        faq_questions: Optional[List[str]],
        goal_id: str
    ) -> None:
        """Publish strategy results to the blackboard."""
        
        results = {}
        
        if competitor_data:
            self.post_to_blackboard(
                "competitor_data",
                competitor_data,
                tags={"strategy", "competitor"}
            )
            results["competitor"] = competitor_data
            self.logger.info("Published competitor data to blackboard")
            
        if faq_questions:
            self.post_to_blackboard(
                "faq_questions",
                faq_questions,
                tags={"strategy", "faq"}
            )
            results["faq_questions"] = faq_questions
            self.logger.info("Published FAQ questions to blackboard")
            
        # Post combined result
        self.post_to_blackboard(
            f"result_{goal_id}",
            results,
            tags={"result"}
        )
        
    def on_message(self, message: Message) -> None:
        """Handle custom messages."""
        self.logger.debug(f"Received message type: {message.type}")
        
        # Handle data responses (e.g., from parser agent)
        if message.type == MessageType.DATA_RESPONSE:
            data_key = message.content.get("data_key")
            if data_key == "product_data":
                # Store for later use
                self.memory.record_observation({
                    "type": "received_product_data",
                    "from": message.sender
                })
                
    def on_task_request(
        self,
        task_name: str,
        params: Dict[str, Any],
        message: Message
    ) -> None:
        """
        Handle task requests from other agents.
        
        Enables collaboration by providing strategy services.
        """
        if task_name == "generate_competitor":
            product_data = params.get("product_data")
            if product_data:
                if isinstance(product_data, dict):
                    product_data = ProductData(**product_data)
                    
                competitor = self._generate_competitor(product_data)
                if competitor:
                    self.send_message(
                        MessageType.DATA_RESPONSE,
                        message.sender,
                        {
                            "task": task_name,
                            "result": competitor.model_dump(),
                            "success": True
                        }
                    )
                    return
                    
        elif task_name == "generate_faqs":
            product_data = params.get("product_data")
            if product_data:
                if isinstance(product_data, dict):
                    product_data = ProductData(**product_data)
                    
                questions = self._generate_faqs(product_data)
                if questions:
                    self.send_message(
                        MessageType.DATA_RESPONSE,
                        message.sender,
                        {
                            "task": task_name,
                            "result": questions,
                            "success": True
                        }
                    )
                    return
                    
        # Send failure response
        self.send_message(
            MessageType.TASK_FAILED,
            message.sender,
            {
                "task": task_name,
                "reason": "Failed to execute strategic task"
            }
        )


# Backwards compatibility - class wrapper
class LegacyStrategyAgent:
    """Legacy wrapper for backwards compatibility."""
    
    def __init__(self):
        self._agent = StrategyAgent()
        logger.debug("StrategyAgent initialized (legacy mode)")
        
    def generate_competitor(self, product: ProductData) -> CompetitorData:
        return self._agent._generate_competitor(product)
        
    def generate_faqs_concepts(self, product: ProductData) -> List[str]:
        return self._agent._generate_faqs(product)
