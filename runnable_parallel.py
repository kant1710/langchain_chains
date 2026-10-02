from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel

load_dotenv()

report_prompt = PromptTemplate(
    template='Write a short report about {topic}',
    input_variables=['topic']
)

post_prompt = PromptTemplate(
    template='Generate a Linkedin post about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'report': RunnableSequence(report_prompt, model, parser),
    'linkedin_post': RunnableSequence(post_prompt, model, parser)
})

result = parallel_chain.invoke({'topic':'Hollywood'})

print(result['report'])
print(result['linkedin_post'])
