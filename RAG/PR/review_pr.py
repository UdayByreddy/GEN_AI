import requests
import os

from dotenv import load_dotenv

load_dotenv()

from langchain_mistralai import ChatMistralAI

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate

PR_NUMBER = 154

OWNER = os.getenv("OWNER")
REPO = os.getenv("REPO")
TOKEN = os.getenv("GIT_TOKEN")



url = f"https://api.github.com/repos/{OWNER}/{REPO}/pulls/{PR_NUMBER}"

headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github.v3.diff"
}

response = requests.get(url, headers=headers)

diff = response.text

prompt  = ChatPromptTemplate.from_messages([
    ('system',"""
You are a senior software engineer and expert code reviewer.

Your task is to review the following pull request diff and generate ONLY production-ready review comments.

IMPORTANT RULES:
1. Give comments ONLY for real issues.
2. Mention the exact file name and line/context.
3. Write comments exactly like a GitHub PR review comment.
4. Keep comments short, professional, and actionable.
5. Do NOT explain outside the comment.
6. Do NOT summarize the PR.
7. Do NOT use markdown headings.
8. Output should be directly copy-pasteable into GitHub review comments.
9. If there are no issues, return exactly:
   "No major issues found."

Review Areas:
- Bugs
- Edge cases
- Null/undefined handling
- Performance
- Security
- React best practices
- python best practices
- Java/Spring Boot best practices
- Clean code
- Naming
- Async issues
- Memory leaks
- API handling
- State management
- Error handling

Output Format:

[file_path]

Comment:
"your review comment"

Code Context:
"small related code snippet"

Severity:
 MEDIUM | HIGH

PR Diff:
{diff}
"""
    
)])

final_prmopt = prompt.invoke({"diff": diff})

# model = ChatMistralAI(
#     model= "mistral-small-2506",
#     temperature=0.9,
# )

model = ChatGroq(
     model="llama-3.3-70b-versatile"
)

response = model.invoke(final_prmopt)


print(response.content)