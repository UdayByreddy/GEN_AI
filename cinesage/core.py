from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser


model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=0.5
)

class MovieReview(BaseModel):
    title: str
    release_year: int
    review: str
    genre: Optional[List[str]] = None
    sentiment: Optional[str] = None
    rating: Optional[str] = None
    strengths: Optional[List[str]] = None
    weaknesses: Optional[List[str]] = None
    rating_mentioned: Optional[str] = None
    
parser = PydanticOutputParser(pydantic_object=MovieReview)

prompt = ChatPromptTemplate.from_messages([('system',"""
                                           Extract important insights from a movie review. 
                                           
                                           {format_rules}"""),(
                                               'human',"""
                                               {review_text}"""
                                           )])


# prompt = ChatPromptTemplate.from_messages([
#     (
#         "system",
#         """
# You are an expert movie review analyzer.

# Your task is to read the given movie review passage carefully
# and convert it into a structured summary.

# Rules:
# - Extract only important insights.
# - Keep points concise and clear.
# - Avoid copying full sentences from the review.
# - Identify sentiment accurately.
# - If information is missing, mention "Not Mentioned".

# Return output in the following format:

# Movie Name:
# Genre:
# Overall Sentiment:

# Summary Points:
# - Point 1
# - Point 2
# - Point 3

# Acting Performance:
# - Lead actor performance
# - Supporting cast performance

# Story & Screenplay:
# - Story quality
# - Screenplay pacing
# - Dialogues

# Direction & Cinematography:
# - Direction quality
# - Visual presentation
# - Camera work

# Music & Background Score:
# - Songs
# - Background music impact

# Strengths:
# - Strength 1
# - Strength 2

# Weaknesses:
# - Weakness 1
# - Weakness 2

# Final Verdict:
# - Short concluding statement

# Rating Mentioned:
# - Extract rating if available
# """
#     ),
#     (
#         "human",
#         """
# Analyze the following movie review and summarize it.

# Movie Review:
# {review_text}
# """
#     )
# ])
para = input("Enter the movie review: ")

final_prompt = prompt.invoke(
    {"review_text": para,
     "format_rules": parser.get_format_instructions()
    }
)
reponse = model.invoke(final_prompt)

print(reponse.content)