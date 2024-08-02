from langchain_core.prompts import ChatPromptTemplate

translation_prompt_template = """
You are an assistant for Korean-English translation tasks.

I will give you the sentence.
If the sentence is already written in English, just copy the sentence.
If not, please translate the sentence from Korean to English.

You should say only the translation of the sentence, and do not say any additional information.

<sentence>
{sentence}
</sentence>

Translation:
"""
translation_prompt = ChatPromptTemplate.from_template(translation_prompt_template)

rewrite_prompt_template = """
You are an assistant for question-revision tasks.
Using given chat history, rephrase the following question to be a standalone question.
The standalone question must have main words of the original question.
Write the revised question in {lang}.

<chat-history>
{history}
</chat-history>

<question>
{query}
</question>

Revised question:
"""
rewrite_prompt = ChatPromptTemplate.from_template(rewrite_prompt_template).partial(lang="English")

expansion_prompt_template = """
Your task is to expand the given query, considering the chat history.
Generate {n} queries that are related to the given query and chat history.

You should provide the queries in {lang}.

All the queries should be separated by a newline.
Do not include any additional information. Only provide the queries.

<chat-history>
{history}
</chat-history>

<question>
{query}
</question>

Queries:
"""
expansion_prompt = ChatPromptTemplate.from_template(expansion_prompt_template).partial(n=3, lang="English")

# restrice the number of sentences to 3, to improve response latency
hyde_prompt_template = """
You are an assistant for question-answering tasks.
Please write a passage to answer the question, considering the given chat history.
Even though you cannot find the context in the chat history, you should generate a passage to answer the question.
Write the answer in {lang}.

Use up to {n} sentences to answer the question.

<chat-history>
{history}
</chat-history>

<question>
{query}
</question>

Answer:
"""
hyde_prompt = ChatPromptTemplate.from_template(hyde_prompt_template).partial(n=3, lang="English")

generation_with_hierarchy_prompt_template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question considering the chat history.

I have provided you with the base context and additional context.
Base context is the main context that you should consider first.
It is the most base information that contains the answer.
Additional context contains information about changes or updates to the base context, 
which are made according to the version updates of the documents or by the request of the specific clients.

You should think step by step.
You should first find the answer from the base context,
and then consider the additional context to see if there are any updates or changes to the answer.
If there are several changes, you should consider the additional context of the most recent version to provide the most up-to-date answer, also comparing the changes to the base context.

If you don't know the answer, just say that you don't know.

You can answer in descriptive form or paraphrased form if you want, and keep the answer concise.

You should answer with the format of the example answer below.
When you reference the documents, you should provide the exact title of the document.
Feel free to use markdown to format your answer.

--------------------------------------------------
**** Example 1 ****
<chat-history>
Human: Can you describe the features of UTIL-1?
</chat-history>

<context>
    <base-context>
    --- Document: Datacenter NVMe SSD Specification v2.0r21.pdf ---
    Average Score: 0.99
    DOC META:
    {{'doc_id': 's3://llm-project-demo-bucket/frequently_access_documents/OCP/2.0/Datacenter+NVMe+SSD+Specification+v2.0r21.pdf', 'doc_name': 'Datacenter NVMe SSD Specification v2.0r21.pdf', 'category': 'base', 'version': 'v2.0r21', 'uri': 'https://llm-project-demo-bucket.s3.ap-northeast-1.amazonaws.com/frequently_access_documents/OCP/2.0/Datacenter+NVMe+SSD+Specification+v2.0r21.pdf'}}
    
    --- Chunk: dbc8ea71-dd6e-45df-a78c-0bec284d170c-9869493d-2d77-484c-82c3-c9b4befcef55 ---
    Score: 0.99
    TEXT:
    16.1 NVMe CLI Management Utility
    The NVMeCLI utility (https://github.com/linux-nvme/nvme-cli) shall be used as one of the management utilities for NVMe devices.
    Requirement ID: UTIL-1
    Description: The SSD supplier must test their SSDs with this utility and ensure compatibility. The following is the minimum list of commands that need to be tested with NVMeCLI:
    - Format.
    - Secure erase.
    - FW update.
    - Controller reset to load FW.
    - Health status.
    - Log page reads including vendor log pages.
    - SMART status.
    CHUNK META:
    {{'chunk_id': 'dbc8ea71-dd6e-45df-a78c-0bec284d170c-9869493d-2d77-484c-82c3-c9b4befcef55', 'page': 110, 'score': 'HIGH'}}
    

    --- Document: datacenter-nvme-ssd-specification-v2-5-pdf.pdf ---
    Average Score: 0.97
    DOC META:
    {{'doc_id': 's3://llm-project-demo-bucket/frequently_access_documents/OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf', 'doc_name': 'datacenter-nvme-ssd-specification-v2-5-pdf.pdf', 'category': 'base', 'version': 'v2.5', 'uri': 'https://llm-project-demo-bucket.s3.ap-northeast-1.amazonaws.com/frequently_access_documents/OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf'}}
    
    --- Chunk: 7b1b1b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b-7b7b7b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b ---
    Score: 0.97
    TEXT:
    18.1 NVMe CLI Management Utility
    The NVMeCLI utility (https://github.com/linux-nvme/nvme-cli/tree/master/plugins/ocp) shall be used as one of the management utilities for NVMe devices.
    Requirement ID: UTIL-1
    Description: The SSD supplier must test their SSDs with this utility and ensure compatibility. The following is the minimum list of commands that need to be tested with NVMeCLI:
    - Format.
    - Secure erase.
    - FW update.
    - Controller reset to load FW.
    - Health status.
    - Log page reads including vendor log pages.
    - SMART status.
    CHUNK META:
    {{'chunk_id': '7b1b1b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b-7b7b7b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b', 'page': 170, 'score': 'HIGH'}}

    </base-context>
    <additional-context>
    --- Document: Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf ---
    Based on: s3://llm-project-demo-bucket/frequently_access_documents/OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf
    Average Score: 0.89
    DOC META:
    {{'doc_id': 's3://llm-project-demo-bucket/frequently_access_documents/OCP/2.5/Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf', 'doc_name': 'Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf', 'category': 'additional', 'version': 'v2.5-addendum-v0.20', 'uri': 'https://llm-project-demo-bucket.s3.ap-northeast-1.amazonaws.com/frequently_access_documents/OCP/2.5/Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf', 'base_doc_id': 's3://llm-project-demo-bucket/frequently_access_documents/OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf'}}
    
    --- Chunk: 7b1b1b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b-7b7b7b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b ---
    Score: 0.89
    TEXT:
    2. Change List
    2.1. Utilization Features
    Requirement ID: UTIL-1
    Description: The list of commands to be tested has been updated.
    - Format.
    - Secure erase.
    - FW update.
    - Controller reset to load FW.
    - Health status.
    - SMART status.
    CHUNK META:
    {{'chunk_id': '7b1b1b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b-7b7b7b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b', 'page': 3, 'score': 'HIGH'}}
    </additional-context>
</context>

<question>
Can you describe the features of UTIL-1?
</question>

Answer:
### Answer
I can find the requirements for UTIL-1 in the document "Datacenter NVMe SSD Specification v2.0r21.pdf" and "datacenter-nvme-ssd-specification-v2-5-pdf.pdf". 
UTIL-1 describes that the ssd supplier must test their SSDs with the following utility and ensure compatibility. The followings are the minimum list of commands that should be tested with NVMeCLI.

- Format
- Secure erase
- FW update
- Controller reset to load FW
- Health status
- SMART status.
If you want to know more details, you can refer to the document "datacenter-nvme-ssd-specification-v2-5-pdf.pdf" page 170, which is the latest version of the document.

### Changes
The requirements for UTIL-1 is consistent in the documents between version 2.0r21 and 2.5.
The only difference is the section number, which is 16.1 in the document "Datacenter NVMe SSD Specification v2.0r21.pdf" and 18.1 in the document "datacenter-nvme-ssd-specification-v2-5-pdf.pdf".

However, the document "Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf" provides an updated list of commands to be tested with NVMeCLI.
While the list of commands in the base document(datacenter-nvme-ssd-specification-v2-5-pdf.pdf) contains the command **Log page reads including vendor log pages**, it is removed in the updated list of commands in the addendum document.

### References
#### Base Documents
- Datacenter NVMe SSD Specification v2.0r21.pdf, page 110
- datacenter-nvme-ssd-specification-v2-5-pdf.pdf, page 170

#### Additional Documents
- Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20, page 3

--------------------------------------------------

<chat-history>
{history}
</chat-history>

<context>
{context}
</context>

<question>
{query}
</question>

Answer:
"""
generation_with_hierarchy_prompt = ChatPromptTemplate.from_template(generation_with_hierarchy_prompt_template)

generation_without_hierarchy_prompt_template = """
You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question considering the chat history.

If you don't know the answer, just say that you don't know.

You can answer in descriptive form or paraphrased form if you want, and keep the answer concise.

You should answer with the format of the example answer below.
Feel free to use markdown to format your answer.

--------------------------------------------------
**** Example 1 ****
<chat-history>
Human: Can you describe the features of UTIL-1?
</chat-history>

<context>
--- Document: Datacenter NVMe SSD Specification v2.0r21.pdf ---
Average Score: 0.99
DOC META:
{{'doc_id': 's3://llm-project-demo-bucket/frequently_access_documents/OCP/2.0/Datacenter+NVMe+SSD+Specification+v2.0r21.pdf', 'doc_name': 'Datacenter NVMe SSD Specification v2.0r21.pdf', 'category': 'base', 'version': 'v2.0r21', 'uri': 'https://llm-project-demo-bucket.s3.ap-northeast-1.amazonaws.com/frequently_access_documents/OCP/2.0/Datacenter+NVMe+SSD+Specification+v2.0r21.pdf'}}

--- Chunk: dbc8ea71-dd6e-45df-a78c-0bec284d170c-9869493d-2d77-484c-82c3-c9b4befcef55 ---
Score: 0.99
TEXT:
16.1 NVMe CLI Management Utility
The NVMeCLI utility (https://github.com/linux-nvme/nvme-cli) shall be used as one of the management utilities for NVMe devices.
Requirement ID: UTIL-1
Description: The SSD supplier must test their SSDs with this utility and ensure compatibility. The following is the minimum list of commands that need to be tested with NVMeCLI:
- Format.
- Secure erase.
- FW update.
- Controller reset to load FW.
- Health status.
- Log page reads including vendor log pages.
- SMART status.
CHUNK META:
{{'chunk_id': 'dbc8ea71-dd6e-45df-a78c-0bec284d170c-9869493d-2d77-484c-82c3-c9b4befcef55', 'page': 110, 'score': 'HIGH'}}


--- Document: datacenter-nvme-ssd-specification-v2-5-pdf.pdf ---
Average Score: 0.97
DOC META:
{{'doc_id': 's3://llm-project-demo-bucket/frequently_access_documents/OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf', 'doc_name': 'datacenter-nvme-ssd-specification-v2-5-pdf.pdf', 'category': 'base', 'version': 'v2.5', 'uri': 'https://llm-project-demo-bucket.s3.ap-northeast-1.amazonaws.com/frequently_access_documents/OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf'}}

--- Chunk: 7b1b1b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b-7b7b7b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b ---
Score: 0.97
TEXT:
18.1 NVMe CLI Management Utility
The NVMeCLI utility (https://github.com/linux-nvme/nvme-cli/tree/master/plugins/ocp) shall be used as one of the management utilities for NVMe devices.
Requirement ID: UTIL-1
Description: The SSD supplier must test their SSDs with this utility and ensure compatibility. The following is the minimum list of commands that need to be tested with NVMeCLI:
- Format.
- Secure erase.
- FW update.
- Controller reset to load FW.
- Health status.
- Log page reads including vendor log pages.
- SMART status.
CHUNK META:
{{'chunk_id': '7b1b1b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b-7b7b7b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b', 'page': 170, 'score': 'HIGH'}}

--- Document: Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf ---
Based on: s3://llm-project-demo-bucket/frequently_access_documents/OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf
Average Score: 0.89
DOC META:
{{'doc_id': 's3://llm-project-demo-bucket/frequently_access_documents/OCP/2.5/Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf', 'doc_name': 'Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf', 'category': 'additional', 'version': 'v2.5-addendum-v0.20', 'uri': 'https://llm-project-demo-bucket.s3.ap-northeast-1.amazonaws.com/frequently_access_documents/OCP/2.5/Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf', 'base_doc_id': 's3://llm-project-demo-bucket/frequently_access_documents/OCP/2.5/datacenter-nvme-ssd-specification-v2-5-pdf.pdf'}}

--- Chunk: 7b1b1b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b-7b7b7b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b ---
Score: 0.89
TEXT:
2. Change List
2.1. Utilization Features
Requirement ID: UTIL-1
Description: The list of commands to be tested has been updated.
- Format.
- Secure erase.
- FW update.
- Controller reset to load FW.
- Health status.
- SMART status.
CHUNK META:
{{'chunk_id': '7b1b1b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b-7b7b7b7b-7b7b-4b7b-8b7b-7b7b7b7b7b7b', 'page': 3, 'score': 'HIGH'}}
</context>

<question>
Can you describe the features of UTIL-1?
</question>

Answer:
### Answer
I can find the requirements for UTIL-1 in the document "Datacenter NVMe SSD Specification v2.0r21.pdf" and "datacenter-nvme-ssd-specification-v2-5-pdf.pdf". 
UTIL-1 describes that the ssd supplier must test their SSDs with the following utility and ensure compatibility. The followings are the minimum list of commands that should be tested with NVMeCLI.

- Format
- Secure erase
- FW update
- Controller reset to load FW
- Health status
- SMART status.
If you want to know more details, you can refer to the document "datacenter-nvme-ssd-specification-v2-5-pdf.pdf" page 170, which is the latest version of the document.

### Changes
The requirements for UTIL-1 is consistent in the documents between version 2.0r21 and 2.5.
The only difference is the section number, which is 16.1 in the document "Datacenter NVMe SSD Specification v2.0r21.pdf" and 18.1 in the document "datacenter-nvme-ssd-specification-v2-5-pdf.pdf".

However, the document "Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20.pdf" provides an updated list of commands to be tested with NVMeCLI.
While the list of commands in the base document(datacenter-nvme-ssd-specification-v2-5-pdf.pdf) contains the command **Log page reads including vendor log pages**, it is removed in the updated list of commands in the addendum document.

### References
- Datacenter NVMe SSD Specification v2.0r21.pdf, page 110
- datacenter-nvme-ssd-specification-v2-5-pdf.pdf, page 170
- Datacenter_NVMe_SSD_Specification_v2.5_Addendum_v0.20, page 3

--------------------------------------------------

<chat-history>
{history}
</chat-history>

<context>
{context}
</context>

<question>
{query}
</question>

Answer:
"""
generation_without_hierarchy_prompt = ChatPromptTemplate.from_template(generation_without_hierarchy_prompt_template)

verification_prompt_template = """
Given context, verify the fact in the response. If the response is correct, say "Yes". If not, say "No".
Context: {context}
Answer: {response}

Verification:
"""
verification_prompt = ChatPromptTemplate.from_template(verification_prompt_template)