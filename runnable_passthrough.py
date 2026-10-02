from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough

load_dotenv()

report_prompt = PromptTemplate(
    template='Write a short report about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

explain_prompt = PromptTemplate(
    template='Explain the following report - {text}',
    input_variables=['text']
)

report_chain = RunnableSequence(report_prompt, model, parser)

parallel_chain = RunnableParallel({
    'report': RunnablePassthrough(),
    'explanation': RunnableSequence(explain_prompt, model, parser)
})

final_chain = RunnableSequence(report_chain, parallel_chain)

print(final_chain.invoke({'topic':'Hollywood'}))