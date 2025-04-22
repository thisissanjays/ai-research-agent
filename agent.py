import operator
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import END, MessagesState, START, StateGraph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string


### LLM

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) 

### Schema 
###chief editor has list of interests of the user, 

class Reporter(BaseModel):
    name: str = Field(
        description="The name of the user of the application.",
    )
    affiliation: str = Field(
        description="Primary affiliation of the reporter.",
    )
    role: str = Field(
        description="The role of the Reporter."
    )
    description: str = Field(
        description= "Description of the Reporter's focus, concerns, and motives.",
    )
    
    @property
    def persona(self) -> str:
        return f"name: {self.name}\ntitle: {self.affiliation}\nrole: {self.role}\ndescription: {self.description}\n"

"""
class newscard_generator(BaseModel):
    topic: str = Field(
        description="The topic/category of the new article.",
    )
    title: str = Field(
        description="The catchy title of the new article.",
    )
    short_news: str = Field(
        description= "The basic summary of the news article to give the user a brief idea of the news",
    )
    
    @property
    def news_list(self) -> str:
        return f"topic: {self.topic}\ntitle: {self.title}\nshort_news: {self.short_news}\n"
"""
class Perspectives(BaseModel):
    reporters : List[Reporter]= Field(
        description="Comprehensive list of Reporters with their roles and affiliations.",
    )

class GenerateReporterstate(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    reporters: List[Reporter] # Analyst asking questions

reporter_instructions="""You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}
        
2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
        
{human_analyst_feedback}
    
3. Determine the most interesting themes based upon documents and / or feedback above.
                    
4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""

def create_analysts(state: GenerateReporterstate):
    
    """ Create analysts """
    
    topic=state['topic']
    max_analysts=state['max_analysts']
    human_analyst_feedback=state.get('human_analyst_feedback', '')
        
    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = reporter_instructions.format(topic=topic,
                                                            human_analyst_feedback=human_analyst_feedback, 
                                                            max_analysts=max_analysts)

    # Generate question 
    reporters = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")])
    
    # Write the list of analysis to state
    return {"reporters": reporters.reporters}




builder = StateGraph(GenerateReporterstate)
builder.add_node("create_analysts" , create_analysts)

builder.add_edge(START, "create_analysts")
builder.add_edge("create_analysts", END)

graph = builder.compile()


