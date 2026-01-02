"""
Builder Agent - Autonomous Content Assembly Agent
=================================================
An autonomous agent responsible for building and assembling
final content pages from processed data.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from src.core.base_agent import BaseAgent, AgentCapability, AgentGoal, AgentState
from src.core.messages import Message, MessageType
from src.models.internal import ProductData, CompetitorData
from src.models.output import FAQPage, ProductPage, ComparisonPage, FAQItem, ComparisonRow
from src.utils.llm_client import get_structured_data
from src.utils.logger import get_agent_logger

# Import Logic Blocks
from src.blocks.benefits import generate_benefits_block
from src.blocks.usage import extract_usage_block
from src.blocks.comparison import compare_products_block

logger = get_agent_logger("BuilderAgent")


class BuilderAgent(BaseAgent):
    """
    Autonomous agent for building content pages.
    
    Capabilities:
    - build_product_page: Assemble product page from data
    - build_faq_page: Generate FAQ page with accurate answers
    - build_comparison_page: Create comparison page between products
    
    The BuilderAgent orchestrates logic blocks and LLM calls
    to assemble final content outputs.
    """
    
    def __init__(self):
        super().__init__(
            agent_id="builder_agent",
            name="Builder Agent",
            description="Autonomous agent for assembling content pages",
            capabilities=[
                AgentCapability(
                    name="build_product_page",
                    description="Build a product page with benefits, usage, and ingredients",
                    input_types=["ProductData"],
                    output_types=["ProductPage"]
                ),
                AgentCapability(
                    name="build_faq_page",
                    description="Build FAQ page with accurate answers",
                    input_types=["ProductData", "List[str]"],
                    output_types=["FAQPage"]
                ),
                AgentCapability(
                    name="build_comparison_page",
                    description="Build comparison page between our product and competitor",
                    input_types=["ProductData", "CompetitorData"],
                    output_types=["ComparisonPage"]
                )
            ]
        )
        
    def plan(self, goal: AgentGoal) -> Optional[List[Dict[str, Any]]]:
        """
        Create a build plan based on the goal.
        
        The agent determines what pages need to be built
        and what data is required.
        """
        self.logger.info(f"Planning build for goal: {goal.description}")
        
        context = goal.context
        plan = []
        
        # Step 1: Acquire all required data
        plan.append({
            "action": "acquire_data",
            "params": {}
        })
        
        # Decision: What to build?
        build_type = context.get("build_type", "all")
        
        if build_type == "product_page" or build_type == "all":
            self.memory.record_decision(
                "Build product page",
                "Product page required for output"
            )
            plan.append({
                "action": "build_product_page",
                "params": {}
            })
            
        if build_type == "faq_page" or build_type == "all":
            self.memory.record_decision(
                "Build FAQ page",
                "FAQ page required for output"
            )
            plan.append({
                "action": "build_faq_page",
                "params": {}
            })
            
        if build_type == "comparison_page" or build_type == "all":
            self.memory.record_decision(
                "Build comparison page",
                "Comparison page required for output"
            )
            plan.append({
                "action": "build_comparison_page",
                "params": {}
            })
            
        # Final step: Publish all results
        plan.append({
            "action": "publish_results",
            "params": {"goal_id": goal.id}
        })
        
        self.logger.info(f"Build plan created with {len(plan)} steps")
        return plan
        
    def execute(self, plan: List[Dict[str, Any]], goal: AgentGoal) -> bool:
        """Execute the build plan."""
        
        # Data containers
        product_data = None
        competitor_data = None
        faq_questions = None
        
        # Output containers
        product_page = None
        faq_page = None
        comparison_page = None
        
        for step in plan:
            action = step["action"]
            params = step["params"]
            
            self.logger.debug(f"Executing build step: {action}")
            
            try:
                if action == "acquire_data":
                    # Get product data from blackboard
                    product_data = self.read_from_blackboard("product_data")
                    if not product_data:
                        self.logger.error("Product data not found on blackboard")
                        self.request_assistance("Need product data", "parse_raw_data")
                        return False
                        
                    # Get competitor data
                    competitor_data = self.read_from_blackboard("competitor_data")
                    
                    # Get FAQ questions
                    faq_questions = self.read_from_blackboard("faq_questions")
                    
                    self.memory.record_outcome(action, True, f"Acquired data for {product_data.name}")
                    
                elif action == "build_product_page":
                    product_page = self._build_product_page(product_data)
                    if not product_page:
                        self.memory.record_outcome(action, False, "Failed to build product page")
                        return False
                    self.memory.record_outcome(action, True)
                    
                elif action == "build_faq_page":
                    if not faq_questions:
                        self.logger.warning("No FAQ questions available, requesting...")
                        self.request_assistance("Need FAQ questions", "generate_faqs")
                        self.memory.record_outcome(action, False, "FAQ questions not available")
                        return False
                        
                    faq_page = self._build_faq_page(product_data, faq_questions)
                    if not faq_page:
                        self.memory.record_outcome(action, False, "Failed to build FAQ page")
                        return False
                    self.memory.record_outcome(action, True)
                    
                elif action == "build_comparison_page":
                    if not competitor_data:
                        self.logger.warning("No competitor data available, requesting...")
                        self.request_assistance("Need competitor data", "generate_competitor")
                        self.memory.record_outcome(action, False, "Competitor data not available")
                        return False
                        
                    comparison_page = self._build_comparison_page(product_data, competitor_data)
                    if not comparison_page:
                        self.memory.record_outcome(action, False, "Failed to build comparison page")
                        return False
                    self.memory.record_outcome(action, True)
                    
                elif action == "publish_results":
                    self._publish_results(
                        product_page, faq_page, comparison_page,
                        params.get("goal_id")
                    )
                    self.memory.record_outcome(action, True)
                    
            except Exception as e:
                self.logger.error(f"Build step '{action}' failed: {e}")
                self.memory.record_outcome(action, False, str(e))
                return False
                
        return True
        
    def _build_product_page(self, product: ProductData) -> Optional[ProductPage]:
        """Build the product page using logic blocks."""
        
        self.logger.info(f"Building product page for: {product.name}")
        
        self.memory.record_observation({
            "type": "build_product_page",
            "product": product.name,
            "components": ["benefits", "usage", "ingredients"]
        })
        
        try:
            # 1. Run Logic Blocks
            self.logger.debug("Running BenefitsBlock...")
            marketing_copy = generate_benefits_block(
                product.key_ingredients,
                product.benefits
            )
            self.logger.debug(f"  - Generated {len(marketing_copy)} benefit statements")
            
            self.logger.debug("Running UsageBlock...")
            usage_steps = extract_usage_block(product.usage_instructions)
            self.logger.debug(f"  - Extracted {len(usage_steps)} usage steps")
            
            # 2. Assemble with accurate data
            skin_types = ', '.join(product.skin_type)
            
            page = ProductPage(
                title=product.name,
                price=f"₹{product.price_inr}",
                description=f"A premium {product.concentration} formulation designed for {skin_types} skin types.",
                key_benefits=marketing_copy,
                usage_guide=usage_steps,
                ingredients_list=product.key_ingredients
            )
            
            self.logger.info(f"✓ Product page assembled: {product.name}")
            return page
            
        except Exception as e:
            self.logger.error(f"Product page build failed: {e}")
            return None
            
    def _build_faq_page(
        self,
        product: ProductData,
        questions: List[str]
    ) -> Optional[FAQPage]:
        """Build the FAQ page with accurate answers."""
        
        self.logger.info(f"Building FAQ page for: {product.name}")
        
        self.memory.record_observation({
            "type": "build_faq_page",
            "product": product.name,
            "questions_count": len(questions)
        })
        
        try:
            # Build COMPLETE product context for accurate answers
            complete_product_data = f"""
=== COMPLETE PRODUCT DATA (USE ONLY THIS FOR ANSWERS) ===

PRODUCT NAME: {product.name}
CONCENTRATION: {product.concentration}
PRICE: ₹{product.price_inr}

SUITABLE SKIN TYPES: {', '.join(product.skin_type)}

KEY INGREDIENTS:
{chr(10).join(f'- {ing}' for ing in product.key_ingredients)}

BENEFITS:
{chr(10).join(f'- {ben}' for ben in product.benefits)}

USAGE INSTRUCTIONS: {product.usage_instructions}

SIDE EFFECTS & WARNINGS: {product.side_effects or 'None specified'}

=== END OF PRODUCT DATA ===
"""
            
            selected_qs = questions[:10]  # Use 10 questions for comprehensive FAQ (minimum 5 required)
            
            self.logger.debug(f"Generating accurate answers for {len(selected_qs)} questions...")
            
            prompt = f"""
You are a Product Information Specialist. Your job is to provide 100% ACCURATE answers.

{complete_product_data}

QUESTIONS TO ANSWER:
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(selected_qs))}

STRICT RULES FOR ANSWERING:
1. ONLY use information from the PRODUCT DATA above - NO external knowledge
2. Be DEFINITIVE - Never say "not specified", "may", "might", "could", "it's recommended to consult"
3. If asked about usage, quote the exact instructions from the data
4. If asked about side effects, state them directly as facts
5. If asked about ingredients, list them exactly as provided
6. If asked about price/skin types, state the exact values
7. Categorize each FAQ appropriately: "Usage", "Safety", "Ingredients", "Product Info"
8. Keep answers concise but complete (2-3 sentences max)

ANSWER FORMAT EXAMPLES:
- Q: "How often should I use this?" → A: "Use only in your PM routine. Beginners should start 1-2 times a week, gradually increasing to every other night."
- Q: "What are the side effects?" → A: "Initial side effects include purging, redness, peeling, or dryness during the first 2-4 weeks. If severe burning occurs, wash off immediately."
- Q: "Is it safe during pregnancy?" → A: "No. This product is strictly not recommended for use during pregnancy or breastfeeding."

Return a list of FAQ Items with category, question, and accurate answer.
"""
            
            class FAQList(BaseModel):
                items: List[FAQItem]
                
            result = get_structured_data(prompt, FAQList)
            
            page = FAQPage(
                product_name=product.name,
                faqs=result.items
            )
            
            self.logger.info(f"[OK] FAQ page assembled via LLM: {len(result.items)} Q&As")
            for item in result.items[:3]:
                self.logger.debug(f"  - [{item.category}] {item.question[:40]}...")
            
            return page
            
        except Exception as e:
            self.logger.warning(f"LLM FAQ build failed, using rule-based fallback: {e}")
            return self._build_faq_page_fallback(product, questions)
            
    def _build_faq_page_fallback(
        self,
        product: ProductData,
        questions: List[str]
    ) -> FAQPage:
        """
        Rule-based fallback for FAQ page generation when LLM is unavailable.
        Generates answers based on product data patterns.
        """
        self.logger.info("Using rule-based FAQ page generation")
        
        faqs = []
        
        # Pre-built answer templates based on question patterns
        for q in questions[:10]:  # Limit to 10 FAQs
            q_lower = q.lower()
            
            # Determine category and answer based on question patterns
            if any(kw in q_lower for kw in ["what is", "what are the key", "concentration"]):
                category = "Product Info"
                if "concentration" in q_lower:
                    answer = f"{product.name} contains {product.concentration} of active ingredients for optimal effectiveness."
                elif "ingredient" in q_lower:
                    answer = f"The key ingredients are: {', '.join(product.key_ingredients)}."
                else:
                    answer = f"{product.name} is a {product.concentration} skincare formulation featuring {', '.join(product.key_ingredients[:2])} for {', '.join(product.benefits[:2]).lower()}."
                    
            elif any(kw in q_lower for kw in ["how do i use", "how often", "when", "apply"]):
                category = "Usage"
                answer = f"{product.usage_instructions}"
                
            elif any(kw in q_lower for kw in ["side effect", "safe", "sensitive", "sunscreen"]):
                category = "Safety"
                if product.side_effects:
                    answer = f"{product.side_effects}. Always perform a patch test before first use."
                else:
                    answer = f"This product is formulated for {', '.join(product.skin_type)} skin types. Perform a patch test before first use."
                    
            elif any(kw in q_lower for kw in ["price", "cost", "worth", "investment"]):
                category = "Purchase"
                answer = f"{product.name} is priced at Rs.{product.price_inr}. It offers professional-grade {product.concentration} formulation with {len(product.key_ingredients)} active ingredients."
                
            elif any(kw in q_lower for kw in ["different", "compare", "alternative", "unique"]):
                category = "Comparison"
                answer = f"{product.name} stands out with its {product.concentration} concentration and synergistic blend of {', '.join(product.key_ingredients[:2])} for enhanced {product.benefits[0].lower() if product.benefits else 'results'}."
                
            elif any(kw in q_lower for kw in ["skin type", "oily", "dry", "combination", "suitable"]):
                category = "Skin Type"
                answer = f"{product.name} is specifically formulated for {', '.join(product.skin_type)} skin types."
                
            else:
                category = "Product Info"
                answer = f"{product.name} contains {product.concentration} with benefits including {', '.join(product.benefits[:2]).lower()}."
            
            faqs.append(FAQItem(
                category=category,
                question=q,
                answer=answer
            ))
        
        page = FAQPage(
            product_name=product.name,
            faqs=faqs
        )
        
        self.logger.info(f"[OK] FAQ page assembled via fallback: {len(faqs)} Q&As")
        return page
            
    def _build_comparison_page(
        self,
        product: ProductData,
        competitor: CompetitorData
    ) -> Optional[ComparisonPage]:
        """Build the comparison page."""
        
        self.logger.info(f"Building comparison: {product.name} vs {competitor.name}")
        
        self.memory.record_observation({
            "type": "build_comparison_page",
            "our_product": product.name,
            "competitor": competitor.name
        })
        
        try:
            # Run Comparison Block
            self.logger.debug("Running ComparisonBlock...")
            table_data = compare_products_block(product, competitor)
            self.logger.debug(f"  - Generated {len(table_data)} comparison rows")
            
            # Convert dicts to Pydantic models
            rows = [ComparisonRow(**row) for row in table_data]
            
            # Count wins
            our_wins = sum(1 for row in rows if "Us" in row.verdict)
            their_wins = sum(
                1 for row in rows 
                if "Competitor" in row.verdict and "Us" not in row.verdict
            )
            
            self.logger.debug(f"  - Verdicts: Our product wins {our_wins}, Competitor wins {their_wins}")
            
            # Generate accurate summary based on actual data
            our_advantages = []
            if product.concentration:
                our_advantages.append(f"clinically effective {product.concentration}")
            our_advantages.append(f"{len(product.key_ingredients)} premium active ingredients")
            
            summary = f"{product.name} offers superior value with {', '.join(our_advantages)}, compared to {competitor.name}'s basic formulation."
            
            page = ComparisonPage(
                title=f"{product.name} vs {competitor.name}",
                competitor_name=competitor.name,
                comparison_table=rows,
                summary_verdict=summary
            )
            
            self.logger.info(f"✓ Comparison page assembled: {product.name} vs {competitor.name}")
            
            return page
            
        except Exception as e:
            self.logger.error(f"Comparison page build failed: {e}")
            return None
            
    def _publish_results(
        self,
        product_page: Optional[ProductPage],
        faq_page: Optional[FAQPage],
        comparison_page: Optional[ComparisonPage],
        goal_id: str
    ) -> None:
        """Publish build results to the blackboard."""
        
        results = {}
        
        if product_page:
            self.post_to_blackboard(
                "product_page",
                product_page,
                tags={"output", "product"}
            )
            results["product_page"] = product_page
            self.logger.info("Published product page to blackboard")
            
        if faq_page:
            self.post_to_blackboard(
                "faq_page",
                faq_page,
                tags={"output", "faq"}
            )
            results["faq_page"] = faq_page
            self.logger.info("Published FAQ page to blackboard")
            
        if comparison_page:
            self.post_to_blackboard(
                "comparison_page",
                comparison_page,
                tags={"output", "comparison"}
            )
            results["comparison_page"] = comparison_page
            self.logger.info("Published comparison page to blackboard")
            
        # Post combined result
        self.post_to_blackboard(
            f"result_{goal_id}",
            results,
            tags={"result"}
        )
        
    def on_message(self, message: Message) -> None:
        """Handle custom messages."""
        self.logger.debug(f"Received message type: {message.type}")
        
    def on_task_request(
        self,
        task_name: str,
        params: Dict[str, Any],
        message: Message
    ) -> None:
        """
        Handle task requests from other agents.
        
        Enables collaboration by providing build services.
        """
        product_data = None
        
        # Extract product data from params
        if "product_data" in params:
            pd = params["product_data"]
            product_data = ProductData(**pd) if isinstance(pd, dict) else pd
            
        if task_name == "build_product_page" and product_data:
            result = self._build_product_page(product_data)
            if result:
                self.send_message(
                    MessageType.DATA_RESPONSE,
                    message.sender,
                    {
                        "task": task_name,
                        "result": result.model_dump(),
                        "success": True
                    }
                )
                return
                
        elif task_name == "build_faq_page":
            questions = params.get("questions", [])
            if product_data and questions:
                result = self._build_faq_page(product_data, questions)
                if result:
                    self.send_message(
                        MessageType.DATA_RESPONSE,
                        message.sender,
                        {
                            "task": task_name,
                            "result": result.model_dump(),
                            "success": True
                        }
                    )
                    return
                    
        elif task_name == "build_comparison_page":
            competitor_data = params.get("competitor_data")
            if product_data and competitor_data:
                if isinstance(competitor_data, dict):
                    competitor_data = CompetitorData(**competitor_data)
                    
                result = self._build_comparison_page(product_data, competitor_data)
                if result:
                    self.send_message(
                        MessageType.DATA_RESPONSE,
                        message.sender,
                        {
                            "task": task_name,
                            "result": result.model_dump(),
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
                "reason": "Failed to execute build task"
            }
        )


# Backwards compatibility class
class LegacyBuilderAgent:
    """Legacy wrapper for backwards compatibility."""
    
    def __init__(self):
        self._agent = BuilderAgent()
        logger.debug("BuilderAgent initialized (legacy mode)")
        
    def build_product_page(self, product: ProductData) -> ProductPage:
        return self._agent._build_product_page(product)
        
    def build_faq_page(self, product: ProductData, questions: List[str]) -> FAQPage:
        return self._agent._build_faq_page(product, questions)
        
    def build_comparison_page(
        self,
        product: ProductData,
        competitor: CompetitorData
    ) -> ComparisonPage:
        return self._agent._build_comparison_page(product, competitor)
