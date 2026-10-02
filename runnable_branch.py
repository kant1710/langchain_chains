from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnablePassthrough, RunnableBranch

load_dotenv()

detail_prompt = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

summary_prompt = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

model = ChatOpenAI()

parser = StrOutputParser()

report_gen_chain = detail_prompt | model | parser

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>250, summary_prompt | model | parser),
    RunnablePassthrough()
)

final_chain = RunnableSequence(report_gen_chain, branch_chain)

print(final_chain.invoke({'topic':'OpenAI vs Anthropic'}))


