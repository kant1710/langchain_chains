from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel

load_dotenv()

def word_count(text):
    return len(text.split())

report_prompt = PromptTemplate(
    template='Write a short report on {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

report_chain = RunnableSequence(report_prompt, model, parser)

parallel_chain = RunnableParallel({
    'report': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

final_chain = RunnableSequence(report_chain, parallel_chain)

result = final_chain.invoke({'topic':'Hollywood'})

final_result = """{} \n word count - {}""".format(result['report'], result['word_count'])

print(final_result)